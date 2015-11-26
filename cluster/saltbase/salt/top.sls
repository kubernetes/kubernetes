base:
  '*':
    - base
    - debian-auto-upgrades
    - salt-helpers
{% if grains.get('cloud') == 'aws' %}
    - ntp
{% endif %}
{% if pillar.get('e2e_storage_test_environment', '').lower() == 'true' %}
    - e2e
{% endif %}

  'roles:kubernetes-pool':
    - match: grain
    - docker
{% if pillar.get('network_provider', '').lower() == 'flannel' %}
    - flannel
{% endif %}
    - helpers
    - cadvisor
    - kube-client-tools
    - kube-node-unpacker
    - kubelet
{% if pillar.get('network_provider', '').lower() == 'opencontrail' %}
    - opencontrail-networking-minion
{% else %}
    - kube-proxy
{% endif %}
{% if pillar.get('enable_node_logging', '').lower() == 'true' and pillar['logging_destination'] is defined %}
  {% if pillar['logging_destination'] == 'elasticsearch' %}
    - fluentd-es
  {% elif pillar['logging_destination'] == 'gcp' %}
    - fluentd-gcp
  {% endif %}
{% endif %}
{% if pillar.get('enable_cluster_registry', '').lower() == 'true' %}
    - kube-registry-proxy
{% endif %}
    - logrotate
    - supervisor
{% if pillar.get('network_provider', '').lower() == 'calico' %}
    - calico.node
{% endif %}

  'roles:kubernetes-master':
    - match: grain
    - generate-cert
    - etcd
{% if pillar.get('network_provider', '').lower() == 'flannel' %}
    - flannel-server
    - flannel
{% endif %}
    - kube-apiserver
    - kube-controller-manager
    - kube-scheduler
    - supervisor
{% if grains['cloud'] is defined and not grains.cloud in [ 'aws', 'gce', 'vagrant' ] %}
    - nginx
{% endif %}
    - cadvisor
    - kube-client-tools
    - kube-master-addons
    - kube-admission-controls
{% if pillar.get('enable_node_logging', '').lower() == 'true' and pillar['logging_destination'] is defined %}
  {% if pillar['logging_destination'] == 'elasticsearch' %}
    - fluentd-es
  {% elif pillar['logging_destination'] == 'gcp' %}
    - fluentd-gcp
  {% endif %}
{% endif %}
{% if grains['cloud'] is defined and grains['cloud'] != 'vagrant' %}
    - logrotate
{% endif %}
    - kube-addons
{% if grains['cloud'] is defined and grains['cloud'] in [ 'vagrant', 'gce', 'aws' ] %}
    - docker
    - kubelet
{% endif %}
{% if pillar.get('network_provider', '').lower() == 'opencontrail' %}
    - opencontrail-networking-master
{% elif pillar.get('network_provider', '').lower() == 'calico' %}
    - calico.master
{% endif %}

  'roles:kubernetes-pool-vsphere':
    - match: grain
    - static-routes
