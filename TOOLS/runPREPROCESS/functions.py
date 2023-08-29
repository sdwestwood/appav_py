def read_brain_recorder_vmrk(filename,ecodes):
    
    allevents = []
    start = False
    with open(filename) as f:
        for line in f:
            line = line.strip()
            # TODO: write part that checks line is whats in encodes
            if line[:4] == "Mk1=":
                start = True
            if start:
                line = line.split(",")
                line[1] = line[1].replace('S ','')
                line[1] = line[1].replace('Sync Off','')
                if line[1] != '' and int(line[1]) in ecodes:
                    allevents.append(line[1:3])
    allevents = [list( map(int,i) ) for i in allevents]
    
    return allevents