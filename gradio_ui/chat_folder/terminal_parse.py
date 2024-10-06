import subprocess

def get_ollama_model_list(in_one_line = False, find_starting: str = "llama3.1:latest") -> list:

    NAME, ID, SIZE, MODIFIED = 0, 0, 0, 0

    result = subprocess.run(['ollama', 'list'], stdout=subprocess.PIPE)
    
    result = result.stdout.decode('utf-8')

    out_res = []

    starting_model = ''

    for line in result.split('\n'):
        if (len(line) < 1):
            continue

        if ('failed to get console mode for stdout: The handle is invalid' in line):
            continue

        if ('NAME' in line and MODIFIED == 0):
            NAME = line.rfind('NAME')
            ID = line.rfind('ID')
            SIZE = line.rfind('SIZE')
            MODIFIED = line.rfind('MODIFIED')

            continue

        if (find_starting in line[NAME:ID].strip()):
            starting_model = line[NAME:ID].strip()

        if (in_one_line):
            line_data = f'{line[NAME:ID].strip()} / {line[SIZE:MODIFIED].strip()} / [{line[ID:SIZE].strip()}]'

            if (find_starting in line[NAME:ID].strip()):
                starting_model = line_data

        else:
            line_data = [
                line[NAME:ID].strip(),
                line[ID:SIZE].strip(),
                line[SIZE:MODIFIED].strip(),
                line[MODIFIED:].strip()
            ]

        out_res.append(line_data)


    return out_res, starting_model

def get_ollama_loaded_status() -> list:

    NAME, ID, SIZE, MODIFIED = 0, 0, 0, 0

    result = subprocess.run(['ollama', 'ps'], stdout=subprocess.PIPE)
    
    result = result.stdout.decode('utf-8')

    out_res = []

    starting_model = ''

    for line in result.split('\n'):
        if (len(line) < 1):
            continue

        if ('failed to get console mode for stdout: The handle is invalid' in line):
            continue

        if ('NAME' in line and MODIFIED == 0):
            NAME = line.rfind('NAME')
            ID = line.rfind('ID')
            SIZE = line.rfind('SIZE')
            PROCESSOR = line.rfind('PROCESSOR')
            UNTIL = line.rfind('UNTIL')

            continue

        
        line_data = f'{line[NAME:ID].strip()} / {line[SIZE:PROCESSOR].strip()} / [{line[ID:SIZE].strip()}] | {line[PROCESSOR:UNTIL].strip()}'
        
        return line_data
    

def cmd_ollama_stop_model(model_str):
    subprocess.run(['ollama', 'stop', model_str], stdout=subprocess.PIPE)


if __name__ == '__main__':

    print(get_ollama_model_list())