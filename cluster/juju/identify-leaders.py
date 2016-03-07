#!/usr/bin/python

from subprocess import check_output
import yaml

out = check_output(['juju', 'status', 'kubernetes', '--format=yaml'])
try:
    parsed_output = yaml.safe_load(out)
    model = parsed_output['services']['kubernetes']['units']
    for unit in model:
        if 'workload-status' in model[unit].keys():
            if 'leader' in model[unit]['workload-status']['message']:
                print(unit)
except:
    pass
