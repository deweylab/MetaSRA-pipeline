import time
from sets import Set
import json
from collections import defaultdict, deque

import web
from web import form

import query_metasra

render =  web.template.render('templates/')

urls = (
    '/', 'index',
    '/download', 'download',
    '/publication', 'publication',
    '/links', 'links',
    '/download_search_result', 'download_search_result'
)

search_form = form.Form(
    form.Textbox(
        'term_id', 
        form.notnull,
        description="Ontology term ID: ",
        class_='textfield', 
        id='term_id_field'
    ),
    form.Radio(
        'sample_type',
        [
        ]
    )
)


class publication:
    def GET(self):
        return render.publication()

class download:
    def GET(self):
        versions = [] # TODO gather all versions of the MetaSRA
        return render.download(versions)

class links:
    def GET(self):
        return render.links()

class download_search_result:
    def POST(self):
        form = search_form()
        form.validates()

        usr_in = form['term_id'].value
        sample_type = form["sample_type"].value
        term_ids = get_searched_term_ids(usr_in)

        metasra_db = web.database(dbn='sqlite', db='static/metasra.sqlite')

        tsv_str = "sample_accession\tstudy_accession\tsample_type\n"
        for term_id in term_ids:
            if sample_type == "all":
                results = query_metasra.query_metasra_for_term(metasra_db, term_id)
            else:
                results = query_metasra.query_metasra_for_term(metasra_db, term_id, sample_type=sample_type)

            for r in results:
                sample_acc = r["sample_accession"]
                study_acc = r["study_accession"]
                sample_type = r["sample_type"]
                confidence = r["confidence"]
                tsv_str += "%s\t%s\t%s\t%0.3f\n" % (sample_acc, study_acc, sample_type, confidence)
        tsv_str = tsv_str[:-1] # remove trailing line-break
        return tsv_str 
        

class index:
    def GET(self):
        form = search_form()
        return render.index(form)

        #return render.index() 
        #return render.index('Bob')
        #return "Hello world!"

    def POST(self): 

        form = search_form()
        form.validates()
        
        usr_in = form['term_id'].value

        metasra_db = web.database(dbn='sqlite', db='static/metasra.sqlite')
        
        #term_ancestors_db = web.database(dbn='sqlite', db='static/term_ancestors.sqlite')        

        term_ids = get_searched_term_ids(usr_in)

        request_results = []
        for term_id in term_ids:    
            sample_type = form["sample_type"].value
            if sample_type == "all":
                results = query_metasra.query_metasra_for_term(metasra_db, term_id)
            else:
                results = query_metasra.query_metasra_for_term(metasra_db, term_id, sample_type=sample_type)


            print "Preparing results..."
            for r in results:
                sample_acc = r["sample_accession"]
                study_acc = r["study_accession"]
                if r["study_title"]:
                    study_title = r["study_title"]
                else:
                    study_title = ""
                attrs_elem = r["sample_attributes_html"]
                sample_name = r["sample_name"]
                sample_type = r["sample_type"]
                confidence = r["confidence"]
                
                request_results.append([
                    '<a class="sample_accession_link" target="_blank" href="https://www.ncbi.nlm.nih.gov/biosample/%s">%s</a>' % (sample_acc,sample_acc),
                    '<a class="sample_accession_link" target="_blank" href="https://www.ncbi.nlm.nih.gov/biosample/%s">%s</a>' % (sample_acc,sample_name),
                    '<a class="study_link" target="_blank" href="https://trace.ncbi.nlm.nih.gov/Traces/sra/?study=%s">%s</a>' % (study_acc, study_title),
                    "%s (%0.2f)" % (sample_type, confidence),
                    attrs_elem
                ])
            print "Finished preparing results."

        return json.dumps(request_results)

def get_searched_term_ids(usr_in):
    term_names_db = web.database(dbn='sqlite', db='static/term_names.sqlite')
    usr_in = usr_in.strip()
    results =  term_names_db.query(
        "SELECT term_id FROM term_names WHERE term_name=$term_name", 
        vars={'term_name':usr_in}
    )
    term_ids = Set([r["term_id"].encode('utf-8') for r in results])
    print "Found term IDs for query '%s': %s" % (usr_in, term_ids)

    # If no terms were found check if the input was an ontology term ID
    if len(term_ids) == 0 and len(usr_in.split(":")) == 2:
        pref = usr_in.split(":")[0]
        suff = usr_in.split(":")[1]
        valid_pref = pref in Set(["DOID", "UBERON", "CVCL", "CL", "EFO"])
        valid_suff = suff.isdigit()
        if valid_pref and valid_suff:
            print "User input '%s' can be interpreted as a term ID" % usr_in
            term_ids = Set([usr_in])

    return term_ids
    

 
if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()
