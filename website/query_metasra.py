
QUERY_METASRA_NO_SAMPLE_TYPE = """
    SELECT 
    sample_accession,
    sample_name,
    study_accession, 
    study_title, 
    sample_attributes_html, 
    sample_terms_html,
    sample_terms_csv,
    sample_type,
    confidence 
    FROM 
    mapped_ontology_terms JOIN 
    sample USING (sample_accession) JOIN 
    sample_type USING (sample_accession) 
    WHERE 
    term_id=$term_id
    """

QUERY_METASRA_YES_SAMPLE_TYPE = """
    SELECT 
    sample_accession,
    sample_name, 
    study_accession, 
    study_title, 
    sample_attributes_html,
    sample_terms_html,
    sample_terms_csv, 
    sample_type,
    confidence 
    FROM 
    mapped_ontology_terms JOIN 
    sample USING (sample_accession) 
    JOIN sample_type USING (sample_accession) 
    WHERE 
    term_id=$term_id AND 
    sample_type=$sample_type
    """


def query_metasra_for_term(db_conn, term_id, sample_type=None):
    print "Querying database..."
    if not sample_type:
        results = db_conn.query(
            QUERY_METASRA_NO_SAMPLE_TYPE,
            vars={'term_id':term_id}
        )
    else:
        results = db_conn.query(
            QUERY_METASRA_YES_SAMPLE_TYPE,
            vars={'term_id':term_id, 'sample_type':sample_type}
        )
    print "Finished query."
    return results



