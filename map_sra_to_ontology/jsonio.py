import json
import sys

def dumps(obj):
    s = json.dumps(obj, indent=4, sort_keys=True, separators=(',', ': '))
    if sys.version_info[0] == 2:
        return unicode(s)
    else:
        return str(s)
