base:
  '*':
    - base
    - debian-auto-upgrades

  'roles:kubernetes-pool':
    - match: grain
    - docker
    - kubelet
    - kube-proxy
    - cadvisor
{% if pillar['use-fluentd-es'] is defined and pillar['use-fluentd-es'] %}
    - fluentd-es
{% endif %}
{% if pillar['use-fluentd-gcp'] is defined and pillar['use-fluentd-gcp'] %}
    - fluentd-gcp
{% endif %}
    # We need a binary release of nsinit
    # - nsinit
    - logrotate
{% if grains['cloud'] is defined and grains['cloud'] == 'azure' %}
    - openvpn-client
{% else %}
    - sdn
{% endif %}

  'roles:kubernetes-master':
    - match: grain
    - etcd
    - apiserver
    - controller-manager
    - scheduler
    - nginx
    - kube-client-tools
    - logrotate
{% if grains['cloud'] is defined and grains['cloud'] == 'azure' %}
    - openvpn
{% endif %}

  'roles:kubernetes-pool-vsphere':
    - match: grain
    - static-routes

  'roles:kubernetes-pool-vagrant':
    - match: grain
    - vagrant
