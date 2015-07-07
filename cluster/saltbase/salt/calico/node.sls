{% if grains.network_mode is defined and grains.network_mode == 'calico' %}

include:
  - docker

calicoctl:
  file.managed:
    - name: /home/vagrant/calicoctl
    - source: https://github.com/Metaswitch/calico-docker/releases/download/v0.4.8/calicoctl
    - source_hash: sha512=814fd7369ba395c67e35245115a5885d1722300301d32585f9003f63e94cd26f77e325ae765ba2f6cba2fddec5ffdb8e4f7bc1b326f9dc343cf03e96b77a679e
    - makedirs: True
    - mode: 744

calico-network-plugin:
  file.managed:
    - name: /usr/libexec/kubernetes/kubelet-plugins/net/exec/calico/calico
    - source: https://github.com/Metaswitch/calico-docker/releases/download/v0.4.8/calico_kubernetes
    - source_hash: sha512=36959621229b97bf5660ebbaa27d765b56ce165b2826a48c7942ab39167980bd698ec5956f73897fee7968c1cb7324e4176a78de1895a884abd7c595cda5fd0f
    - makedirs: True
    - mode: 744

calico-node:
  cmd.run:
    - name: /home/vagrant/calicoctl node --ip={{ grains.node_ip }} --node-image=calico/node:v0.4.8
    - env:
      - ETCD_AUTHORITY: "{{ grains.api_servers }}:6666"
    - require:
      - kmod: ip6_tables
      - kmod: xt_set
      - service: docker
      - file: calicoctl

calico-ip-pool-reset:
  cmd.run:
    - name: /home/vagrant/calicoctl pool remove 192.160.0.0/16
    - env:
      - ETCD_AUTHORITY: "{{ grains.api_servers }}:6666"
    - require:
      - service: docker
      - file: calicoctl
      - cmd: calico-node

calico-ip-pool:
  cmd.run:
    - name: /home/vagrant/calicoctl pool add {{ grains['cbr-cidr'] }}
    - env:
      - ETCD_AUTHORITY: "{{ grains.api_servers }}:6666"
    - require:
      - service: docker
      - file: calicoctl
      - cmd: calico-node

ip6_tables:
  kmod.present

xt_set:
  kmod.present

{% endif %}
