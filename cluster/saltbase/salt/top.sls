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
    - helpers
    - cadvisor
    - kubelet
    - kube-proxy
{% if pillar.get('enable_node_logging', '').lower() == 'true' and pillar['logging_destination'] is defined %}
  {% if pillar['logging_destination'] == 'elasticsearch' %}
    - fluentd-es
  {% elif pillar['logging_destination'] == 'gcp' %}
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
{% if grains['cloud'] is defined and not grains.cloud in [ 'aws', 'gce' ] %}
    - nginx
{% endif %}
    - cadvisor
    - kube-client-tools
    - kube-master-addons
{% if grains['cloud'] is defined and grains['cloud'] != 'vagrant' %}
    - logrotate
{% endif %}
    - kube-addons
{% if grains['cloud'] is defined and grains['cloud'] == 'azure' %}
    - openvpn
{% endif %}
{% if grains['cloud'] is defined and grains['cloud'] == 'vagrant' %}
    - docker
    - kubelet
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
