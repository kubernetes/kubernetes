s/__DNS__SERVER__/{{ pillar['dns_server'] }}/g
s/__DNS__DOMAIN__/{{ pillar['dns_domain'] }}/g
s/__CLUSTER_CIDR__/{{ pillar['service_cluster_ip_range'] }}/g
s/__DNS__MEMORY__LIMIT__/{{ pillar['dns_memory_limit'] }}/g
s/__MACHINE_GENERATED_WARNING__/Warning: This is a file generated from the base underscore template file: __SOURCE_FILENAME__/g
