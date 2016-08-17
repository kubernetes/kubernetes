s/__PILLAR__DNS__SERVER__/{{ pillar['dns_server'] }}/g
s/__PILLAR__DNS__REPLICAS__/{{ pillar['dns_replicas'] }}/g
s/__PILLAR__DNS__DOMAIN__/{{ pillar['dns_domain'] }}/g
s/__PILLAR__FEDERATIONS__DOMAIN__MAP__/{{ pillar['federations_domain_map'] }}/g
s/__MACHINE_GENERATED_WARNING__/Warning: This is a file generated from the base underscore template file: __SOURCE_FILENAME__/g