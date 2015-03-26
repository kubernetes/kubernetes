base:
  '*':
    - base
    - debian-auto-upgrades

  'roles:kubernetes-pool':
    - match: grain
    - docker
{% if grains['cloud'] is defined and grains['cloud'] == 'azure' %}
    - openvpn-client
{% else %}
    - sdn
{% endif %}
    - cadvisor
    - kubelet
    - kube-proxy
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

  'roles:kubernetes-master':
    - match: grain
    - generate-cert
    - etcd
    - kube-apiserver
    - kube-controller-manager
    - kube-scheduler
    - monit
    - nginx
    - cadvisor
    - kube-client-tools
{% if grains['cloud'] is defined and grains['cloud'] != 'vagrant' %}
    - logrotate
{% endif %}
    - kube-addons
{% if grains['cloud'] is defined and grains['cloud'] == 'azure' %}
    - openvpn
{% endif %}
{% if grains['cloud'] is defined and grains['cloud'] == 'vagrant' %}
    - docker
    - sdn
{% endif %}
{% if grains['cloud'] is defined and grains['cloud'] == 'aws' %}
    - docker
    - kubelet
{% endif %}
{% if grains['cloud'] is defined and grains['cloud'] == 'gce' %}
    - docker
    - kubelet
{% endif %}


  'roles:kubernetes-pool-vsphere':
    - match: grain
    - static-routes
