import sys
import json

if __name__ == "__main__":

    # Input file
    assert len(sys.argv) == 2
    nbfilein  = sys.argv[1]

    # JSON file
    jsonin = json.load(open(nbfilein, 'rt'))
    cells  = jsonin['cells']
    
    counter = 0
    for cell in cells:
        if 'execution_count' in cell:
            counter += 1
            cell['execution_count'] = counter
        if  'metadata' in cell:
            cell['metadata'] = {}
        
        if 'outputs' in cell:
            for cellout in cell['outputs']:
                if 'execution_count' in cellout:
                    cellout['execution_count'] = counter
                if  'metadata' in cellout:
                    cellout['metadata'] = {}


    json.dump(jsonin, open(nbfilein, 'wt'), 
              sort_keys=True, indent=4, ensure_ascii=False)
