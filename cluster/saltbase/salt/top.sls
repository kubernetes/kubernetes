base:
  '*':
    - base

  'roles:kubernetes-pool':
    - match: grain
    - docker
    - kubelet
    - kube-proxy
    - cadvisor
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

