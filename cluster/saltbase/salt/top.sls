base:
  '*':
    - base
    - debian-auto-upgrades

  'roles:lmktfyrnetes-pool':
    - match: grain
    - docker
{% if grains['cloud'] is defined and grains['cloud'] == 'azure' %}
    - openvpn-client
{% else %}
    - sdn
{% endif %}
    - cadvisor
    - lmktfylet
    - lmktfy-proxy
{% if pillar.get('enable_node_logging', '').lower() == 'true' %}
  {% if pillar['logging_destination'] is defined and pillar['logging_destination'] == 'elasticsearch' %}
    - fluentd-es
  {% endif %}
  {% if pillar['logging_destination'] is defined and pillar['logging_destination'] == 'gcp' %}
    - fluentd-gcp
  {% endif %}
{% endif %}
    - logrotate
    - monit

  'roles:lmktfyrnetes-master':
    - match: grain
    - generate-cert
    - etcd
    - lmktfy-apiserver
    - lmktfy-controller-manager
    - lmktfy-scheduler
    - monit
    - nginx
    - cadvisor
    - lmktfy-client-tools
{% if grains['cloud'] is defined and grains['cloud'] != 'vagrant' %}
    - logrotate
{% endif %}
    - lmktfy-addons
{% if grains['cloud'] is defined and grains['cloud'] == 'azure' %}
    - openvpn
{% endif %}
{% if grains['cloud'] is defined and grains['cloud'] == 'vagrant' %}
    - docker
    - sdn
{% endif %}
{% if grains['cloud'] is defined and grains['cloud'] == 'aws' %}
    - docker
    - lmktfylet
{% endif %}
{% if grains['cloud'] is defined and grains['cloud'] == 'gce' %}
    - docker
    - lmktfylet
{% endif %}


  'roles:lmktfyrnetes-pool-vsphere':
    - match: grain
    - static-routes
