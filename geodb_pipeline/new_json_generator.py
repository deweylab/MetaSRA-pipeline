import sqlite3
import json
import csv
from sqlite3 import Error

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return None

def select_ten_tasks(conn):
    """
    Query 10 rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM gsm limit 10")

    rows = cur.fetchall()

    for row in rows:
        print(row)

def select_json(conn,gpl_list):
    """
    Query rows in the gsm table to get the title, sample and charateristics 
    :param conn: the Connection object
    :return:
    """
    gpl_list_str = ",".join([str(x) for x in gpl_list])
    print(gpl_list_str)
    cur = conn.cursor()
    #gpl_list = gpl_list[1:2]
    gpl_size = 0
    for item in gpl_list:
        print (gpl_size)
        cur.execute('select gsm,title,type,source_name_ch1,characteristics_ch1 from gsm where gpl=?',(item,))
        rows = cur.fetchall()
        count = 0
        shrunk_json = []
        dummy_json = []
        
        for row in rows:
            dummy_json = []
            count2 = 1
            for key in cur.description:
                if(key[0] == 'gsm'): 
                    #print (key)
                    #print("here....")
                    #print(key[0])
                    continue
                #print (type(key[0]))
                #print(key)
                dummy_json.append({key[0]:row[count2]})
                count2=count2+1
            shrunk_json.append({row[0]:dummy_json})
        file_name = 'gpl_files/{}.json'.format(gpl_size)
        with open(file_name, 'w') as outfile:
            d = json.dumps({item:shrunk_json})
            json.dump(json.loads(d), outfile)
        gpl_size = gpl_size+1
        # count = count+1
        # if count==10 :
        #         break
    return 0

def supported_microarray_csv2list(csv_file):
    """
    Get list of external_accession from the supported microarray platforms for human only
    :param file: A csv file
    :return: the list of rows in column 2
    """
    count = 0
    ans = []
    with open(csv_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
                #print(row)
                ans.append(row[1])

    return ans[1:]

def main():
    """
    This is the main function that returns a json object with limited columns for the current list of supported_microarray as of 25 May
    :param 
    :return: json object
    """    
    supported_microarray_csv_file = "supported_microarray.csv"
    supported_microarray_list = supported_microarray_csv2list(supported_microarray_csv_file)
    print(len(supported_microarray_list))
    print(len(set(supported_microarray_list)))
    #for item in supported_microarray_list:
    #   print(item)
    database = "GEOmetadb.2019-04-22.sqlite"

    # create a database connection
    conn = create_connection(database)
    json_content = ""
    with conn:
        print("1. Query 10 tasks")
        #select_ten_tasks(conn)
        select_json(conn,supported_microarray_list)
    # d = json.loads(json_content)
    # print(len(d))
    # print(len(d['shrunk_json']))
    # print((d['shrunk_json'][0]))
    # print(d['shrunk_json'][0]['GSM3463220'])
    # with open('data1.json', 'w') as outfile:
    #     json.dump(d, outfile)
    # return d

if __name__ == '__main__':
    main()

