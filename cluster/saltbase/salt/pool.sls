include:
  - generate-cert
  - etcd
{% if pillar.get('network_provider', '').lower() == 'flannel' %}
  - flannel-server
  - flannel
{% elif pillar.get('network_provider', '').lower() == 'kubenet' %}
  - cni
{% elif pillar.get('network_provider', '').lower() == 'cni' %}
  - cni
{% endif %}
{% if pillar.get('enable_l7_loadbalancing', '').lower() == 'glbc' %}
  - l7-gcp
{% endif %}
  - kube-apiserver
  - kube-controller-manager
  - kube-scheduler
  - supervisor
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
{% if grains['cloud'] is defined and grains['cloud'] in [ 'vagrant', 'gce', 'aws', 'vsphere', 'photon-controller', 'openstack'] %}
  - docker
  - kubelet
{% endif %}
{% if pillar.get('network_provider', '').lower() == 'opencontrail' %}
  - opencontrail-networking-master
{% endif %}
{% if pillar.get('enable_cluster_autoscaler', '').lower() == 'true' %}
  - cluster-autoscaler
{% endif %}
