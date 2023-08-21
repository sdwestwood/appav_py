#This script does not effect the running of the code, but helps with useability.
#There are a few functions which makes life easier

#Used for basic stimcode searching in get_events
basicRead = lambda tmpevents, codes: [[i[1] for code in codes for i in tmpevents if i[0] in code] for _ in range(len(codes))]

#TODO: automatic subject name loading would be good to have hear, file directory thing