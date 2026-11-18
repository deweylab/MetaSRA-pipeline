########################################################################
#
# Run the ontology mapping pipeline on a set of key-value pairs 
# that describe a biological sample
#
########################################################################

from optparse import OptionParser

from metasra.map_sra_to_ontology.run_pipeline import run_pipeline

def main():
    parser = OptionParser()
    #parser.add_option("-f", "--key_value_file", help="JSON file storing key-value pairs describing sample")
    parser.add_option("-r", "--ref_path", default="metasra_ref", 
                      help="Path to where reference files are stored.")
    parser.add_option("-o", "--output", dest="output", default=None,
                      help="Output file to which to write results (default: stdout)")
    parser.add_option("-f", "--fine_grained", dest="fine_grained", default=False,
                      action="store_true", help="Output mappings stratified by explicit and consequent")
    parser.add_option("-c", "--candidate_mentions", dest="candidate_mentions", default=False,
                      action="store_true", help="Output candidate mentions only")
    (options, args) = parser.parse_args()
   
    input_f = args[0]

    run_pipeline(
        input_f, 
        ref_path=options.ref_path, 
        output=options.output, 
        fine_grained=options.fine_grained, 
        candidate_mentions=options.candidate_mentions
    )

if __name__ == "__main__":
    main()
