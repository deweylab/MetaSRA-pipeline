from __future__ import print_function
from optparse import OptionParser
import sqlite3
import collections
from collections import defaultdict

THE_DB_LOC = "/tier2/deweylab/mnbernstein/sra_metadb/SRAmetadb.17-09-15.sqlite"
SUB_DB_LOC = "/tier2/deweylab/mnbernstein/sra_metadb/SRAmetadb.subdb.17-09-15.TEST.sqlite"

def main():
    usage = '%prog <assay> <species>'
    parser = OptionParser()
    parser.add_option("-t", "--the_db", help="Location of the original SRAdb")
    parser.add_option("-s", "--sub_db", help="Location of the 'sub'SRAdb")
    (options, args) = parser.parse_args()

    assay = args[0]
    species = args[1]

    if options.the_db and options.sub_db:
        build_subdb(options.the_db, options.sub_db, assay, species)
    else:
        build_subdb(THE_DB_LOC, SUB_DB_LOC, assay, species)

def build_subdb(the_db_loc, sub_db_loc, assay, species):

    ##### Build experiment table

    drop_experiment_table_sql = """DROP TABLE experiment"""
    drop_sample_table_sql = """DROP TABLE sample"""
    drop_sample_attribute_table_sql = """DROP TABLE sample_attribute"""
    drop_read_spec_table_sql = """DROP TABLE read_spec"""
    drop_run_table_sql = """DROP TABLE run"""
    drop_study_table_sql = """DROP TABLE study"""

    create_experiment_table_sql = """CREATE TABLE experiment (experiment_accession text PRIMARY KEY NOT NULL, 
        title text, design_description text, study_accession text NOT NULL, 
        sample_accession text NOT NULL, library_source text, library_selection text, 
        library_layout text, library_construction_protocol text, spot_length int, instrument_model text,
        submission_accession text, sradb_updated text)"""
 
    create_sample_table_sql = """CREATE TABLE sample (sample_accession text PRIMARY KEY NOT NULL,
        center_name text, description text, submission_accession text, sra_db_updated text, 
        dbgap_accession text)"""

    create_sample_attribute_table_sql = """CREATE TABLE sample_attribute (sample_accession text NOT NULL, 
        tag text NOT NULL, value text NOT NULL, PRIMARY KEY (sample_accession, tag, value))"""

    create_read_spec_table_sql = """CREATE TABLE read_spec (experiment_accession text NOT NULL, 
        read_index int NOT NULL, read_class text NOT NULL, read_type text NOT NULL, base_coord int,
        PRIMARY KEY (experiment_accession, read_index))"""

    create_run_table_sql = """CREATE TABLE run (run_accession text NOT NULL, experiment_accession text NOT NULL, 
        run_date text, submission_accession text, sradb_updated text, 
        PRIMARY KEY (run_accession, experiment_accession))
        """
    create_study_table_sql = """CREATE TABLE study (study_accession text NOT NULL, study_title text, 
        study_abstract text, center_name text, study_description text, 
        submission_accession text, sradb_updated text, PRIMARY KEY (study_accession))"""

    query_sample_sql = """SELECT sample_accession, sample.center_name, sample.description, sample_url_link,
        sample.xref_link, sample_attribute, sample.submission_accession, sample.sradb_updated FROM 
        experiment JOIN sample USING (sample_accession) WHERE library_strategy = '%s' 
        AND scientific_name = '%s' AND platform = 'ILLUMINA'
        """ % (assay, species)

    query_experiment_sql = """SELECT experiment_accession, title, design_description, study_accession, 
        sample_accession, library_source, library_selection, library_layout, library_construction_protocol,
        spot_length, read_spec, instrument_model, experiment_url_link, experiment.xref_link, experiment_attribute,
        experiment.submission_accession, experiment.sradb_updated FROM experiment JOIN sample USING (sample_accession) WHERE 
        library_strategy = '%s' AND scientific_name = '%s' AND platform = 'ILLUMINA'
        """ % (assay, species)

    query_study_sql = """ SELECT study_accession, study_title, study_abstract, center_name, study_description, 
        xref_link, study_attribute, submission_accession, sradb_updated FROM study JOIN (SELECT experiment_accession, 
        study_accession, scientific_name, library_strategy, platform FROM experiment JOIN sample USING (sample_accession)) 
        USING (study_accession) WHERE scientific_name = '%s' AND library_strategy = '%s' 
        AND platform = 'ILLUMINA'""" % (species, assay)

    query_run_sql = """SELECT run_accession, experiment_accession, run_date, submission_accession, sradb_updated
        FROM run JOIN (SELECT experiment_accession, scientific_name, platform, library_strategy FROM experiment 
        JOIN sample USING (sample_accession)) USING (experiment_accession) WHERE library_strategy = '%s' 
        AND scientific_name = '%s' AND platform = 'ILLUMINA'""" % (assay, species)

    insert_update_experiment_sql = """INSERT OR REPLACE INTO experiment VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""" 
    insert_update_sample_sql = """INSERT OR REPLACE INTO sample VALUES (?, ?, ?, ?, ?, ?)"""
    insert_update_sample_attribute_sql = """INSERT OR REPLACE INTO sample_attribute VALUES (?, ?, ?)"""
    insert_update_read_spec_sql = "INSERT OR REPLACE INTO read_spec VALUES (?, ?, ?, ?, ?)"
    insert_update_run_sql = "INSERT OR REPLACE INTO run VALUES (?, ?, ?, ?, ?)"
    insert_update_study_sql = "INSERT OR REPLACE INTO study VALUES (?, ?, ?, ?, ?, ?, ?)"

    with sqlite3.connect(the_db_loc) as the_db_conn:
        with sqlite3.connect(sub_db_loc) as sub_db_conn:
            sub_c = sub_db_conn.cursor()
            the_c = the_db_conn.cursor()
       
            # Drop existing tables, create new table
            try: 
                # Drop existing tables, create new tables
                print("Dropping old 'experiment' table...")
                sub_c.execute(drop_experiment_table_sql)
            except sqlite3.OperationalError as e:
                print(e)
            print("Creating new 'experiment' table...")
            sub_c.execute(create_experiment_table_sql)

            try:
                print("Dropping old 'read_spec' table...")
                sub_c.execute(drop_read_spec_table_sql)
            except sqlite3.OperationalError as e:
                print(e)                
            print("Creating new 'read_spec' table...")
            sub_c.execute(create_read_spec_table_sql)

            try:
                print("Dropping old 'sample' table...")
                sub_c.execute(drop_sample_table_sql)
            except sqlite3.OperationalError as e:
                print(e)
            print("Creating new 'sample' table...")
            sub_c.execute(create_sample_table_sql)

            try:
                print("Dropping old 'sample_attribute' table...")
                sub_c.execute(drop_sample_attribute_table_sql)
            except sqlite3.OperationalError as e:
                print(e)
            print("Creating new 'sample_attribute' table...")
            sub_c.execute(create_sample_attribute_table_sql)


            try:
                print("Dropping old 'run' table...")
                sub_c.execute(drop_run_table_sql)
            except sqlite3.OperationalError as e:
                print(e)
            print("Creating new 'run' table...")
            sub_c.execute(create_run_table_sql)

            try:
                print("Dropping old 'study' table...")
                sub_c.execute(drop_study_table_sql)
            except sqlite3.OperationalError as e:
                print(e)            
            print("Creating new 'study' table...")
            sub_c.execute(create_study_table_sql) 
        

            # Grap sample data from SRAdb
            print("Querying relavent sample records from the SRAdb...")
            sample_data = []
            returned = the_c.execute(query_sample_sql)
            for r in returned:
                row = [x for x in r]
                sam_d = {}
                sam_d["sample_accession"]       = row[0]
                sam_d["center_name"]            = row[1]
                sam_d["description"]            = row[2]
                sam_d["sample_url_link"]        = row[3]
                sam_d["xref_link"]              = row[4]
                sam_d["sample_attribute"]       = row[5]
                sam_d["submission_accession"]   = row[6]
                sam_d["sradb_updated"]          = row[7]
                sample_data.append(sam_d)

            # Parse 'sample_attribute' field to get tag-value pairs
            print("Parsing sample attributes...")
            sample_to_tag_to_val = {}
            for sam_d in sample_data:
                if sam_d["sample_attribute"]:
                    tokens = sam_d["sample_attribute"].encode('utf-8').split("||")
                    tag_to_val = {}
                    for t in tokens:
                        if len(t.split(":")) < 2:
                            continue
                        tag = t.split(":")[0].strip()
                        val = t.split(":")[1].strip()
                        tag_to_val[tag] = val
                    sample_to_tag_to_val[sam_d["sample_accession"]] = tag_to_val 
                    if "gap_accession" in sample_to_tag_to_val[sam_d["sample_accession"]]:
                        sam_d["dbgap_accession"] = sample_to_tag_to_val[sam_d["sample_accession"]]["gap_accession"]
                    else:
                        sam_d["dbgap_accession"] = None
                else:
                    sam_d["dbgap_accession"] = None
           
            # Create sample attribute table
            print("Inserting entries into 'sample_attribute' table...")
            for sam_acc, tag_to_val in sample_to_tag_to_val.iteritems():
                for tag, val in tag_to_val.iteritems():
                    insert_tuple = (sam_acc, tag.decode('utf-8'), val.decode('utf-8'))
                    sub_c.execute(insert_update_sample_attribute_sql, insert_tuple) 

            # Create sample table
            print("Inserting entries into 'sample' table...")
            for sam_d in sample_data:
                insert_tuple = (sam_d["sample_accession"], sam_d["center_name"], 
                    sam_d["description"], sam_d["submission_accession"], 
                    sam_d["sradb_updated"], sam_d["dbgap_accession"])
                sub_c.execute(insert_update_sample_sql, insert_tuple)    
            
            # Grab experiment data from SRAdb
            print("Querying relavent experiment records from the SRAdb...")
            experiment_data = []
            exp_to_tag_to_val = {}
            returned = the_c.execute(query_experiment_sql)
            for r in returned: 
                row = [x for x in r]
                exp_d = {}
                exp_d["experiment_accession"]             = row[0]
                exp_d["title"]                            = row[1]
                exp_d["design_description"]               = row[2]
                exp_d["study_accession"]                  = row[3]
                exp_d["sample_accession"]                 = row[4]
                exp_d["library_source"]                   = row[5]
                exp_d["library_selection"]                = row[6]
                exp_d["library_layout"]                   = parsed_single_paired_end(row[7])
                exp_d["library_construction_protocol"]    = row[8]
                exp_d["spot_length"]                      = row[9]
                exp_d["read_spec"]                        = row[10]
                exp_d["instrument_model"]                 = row[11]
                exp_d["experiment_url_link"]              = row[12]
                exp_d["xref_link"]                        = row[13]
                exp_d["experiment_attribute"]             = row[14]
                exp_d["submission_accession"]             = row[15]
                exp_d["sradb_updated"]                    = row[16]
                experiment_data.append(exp_d)

            # Parse read spec data from SRAdb
            exp_to_read_datas = {}
            for exp_d in experiment_data:
                if not exp_d["read_spec"]:
                    continue
                reads = exp_d["read_spec"].encode('utf-8').split("||")
                exp_to_read_datas[exp_d["experiment_accession"]] = []
                for read in reads:
                    read_data = defaultdict(lambda: None)
                    tokens = read.split(";") 
                    for t in tokens:
                        if len(t.split(":")) < 2:
                            continue
                        tag = t.split(":")[0].strip()
                        val = t.split(":")[1].strip()
                        if tag == "READ_INDEX":
                            read_data["read_index"] = int(val)
                        elif tag == "READ_CLASS":
                            read_data["read_class"] = val
                        elif tag == "READ_TYPE":
                            read_data["read_type"] = val
                        elif tag == "BASE_COORD":
                            read_data["base_coord"] = int(val)
                    exp_to_read_datas[exp_d["experiment_accession"]].append(read_data)

            # Create read-spec table
            print("Inserting entries into 'read_spec' table...")
            for exp_acc, read_datas in exp_to_read_datas.iteritems():
                for read_data in read_datas:
                    insert_tuple = (exp_acc, read_data["read_index"], read_data["read_class"], 
                        read_data["read_type"], read_data["base_coord"])
                    sub_c.execute(insert_update_read_spec_sql, insert_tuple)

            # Create experiment table
            print("Inserting entries into 'experiment' table...")
            for exp_d in experiment_data:
                insert_tuple = (exp_d["experiment_accession"], exp_d["title"],
                    exp_d["design_description"], exp_d["study_accession"], exp_d["sample_accession"],
                    exp_d["library_source"], exp_d["library_selection"], exp_d["library_layout"], 
                    exp_d["library_construction_protocol"], exp_d["spot_length"], exp_d["instrument_model"],
                    exp_d["submission_accession"], exp_d["sradb_updated"])
                sub_c.execute(insert_update_experiment_sql, insert_tuple)                

            # Query run table
            print("Querying relavent run records from the SRAdb...")
            run_data = []
            returned = the_c.execute(query_run_sql)
            for r in returned:
                row = [x for x in r]
                run_d = {}
                run_d["run_accession"]          = row[0]
                run_d["experiment_accession"]   = row[1]
                run_d["run_date"]               = row[2]
                run_d["submission_accession"]   = row[3]
                run_d["sradb_update"]           = row[4]
                run_data.append(run_d)
                
            # Create run table
            print("Inserting entries into 'run' table...")
            for run_d in run_data:
                insert_tuple = (run_d["run_accession"], run_d["experiment_accession"], 
                    run_d["run_date"], run_d["submission_accession"], run_d["sradb_update"])
                sub_c.execute(insert_update_run_sql, insert_tuple)

            # Query study table
            print("Querying relavent study records from the SRAdb...")
            study_data = []
            returned = the_c.execute(query_study_sql)
            for r in returned:
                row = [x for x in r]
                study_d = {}
                study_d["study_accession"]      = row[0]
                study_d["study_title"]          = row[1]
                study_d["study_abstract"]       = row[2]
                study_d["center_name"]          = row[3]
                study_d["study_description"]    = row[4]
                study_d["xref_link"]            = row[5]
                study_d["study_attribute"]      = row[6]
                study_d["submission_accession"] = row[7]
                study_d["sradb_updated"]        = row[8]
                study_data.append(study_d)

            # Create study table
            print("Inserting entries into 'study' table...")
            for study_d in study_data:
                insert_tuple = (study_d["study_accession"], study_d["study_title"],
                    study_d["study_abstract"], study_d["center_name"],
                    study_d["study_description"], study_d["submission_accession"],
                    study_d["sradb_updated"])           
                sub_c.execute(insert_update_study_sql, insert_tuple)
            
def parsed_single_paired_end(raw_str):
    # PAIRED - NOMINAL_LENGTH: 101; |132
    # PAIRED - NOMINAL_LENGTH: 175; NOMINAL_SDEV: 70; |154
    # PAIRED - NOMINAL_SDEV: 0.0E0; NOMINAL_LENGTH: 0; |422
    # PAIRED - NOMINAL_SDEV: 50; NOMINAL_LENGTH: 100; |632
    # PAIRED - NOMINAL_LENGTH: 300; |721
    # PAIRED - |60535
    # SINGLE - |74665
    parsed_str = raw_str.split("-")[0].strip()   
    print(parsed_str)
    return parsed_str


if __name__ == "__main__":
    main()




