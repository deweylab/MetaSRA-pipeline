import json
import sys

from . import mapping_path

def dumps(obj):
    s = json.dumps(obj, indent=4, sort_keys=True, separators=(',', ': '), 
                   default=mapping_path.encode)
    if sys.version_info[0] == 2:
        return unicode(s)
    else:
        return str(s)


def loads(s):
    return json.loads(s, object_hook=mapping_path.decode)
