{% if grains.network_mode is defined and grains.network_mode == 'calico' %}

include:
  - docker

calicoctl:
  file.managed:
    - name: /home/vagrant/calicoctl
    #- source: https://github.com/Metaswitch/calico-docker/releases/download/v0.2.0/calicoctl
    #- source_hash: sha512=ff7b6edb4728efc5c56af48a944584962d84e572453b07bbfc5160f07344d1b075a8cc23f9e81a98079b6931bd1dc5a7432bd1d41078b02c51e27fa9508597e4
    - source: salt://calico/calicoctl
    - source_hash: sha512=85224f630abc8090fdc1868bccda00b3078d08a49b87e6a47cb816599db992014442df9d57a8b395daf9cc478fc9d972e9542bbc307c1059419a9eeb4bcb9017
    - makedirs: True
    - mode: 744

calico-network-plugin:
  file.managed:
    - name: /usr/libexec/kubernetes/kubelet-plugins/net/exec/calico/calico
    - source: salt://calico/calico_network_plugin.py
    - source_hash: sha512=515454093f395535214e8891b0cbb2d4d7ae503c3626be0ce8e556f1f020a4066c7babedf855f63a0ab4aebaffe3c2631852d1e8c40b2c5be9eb9310b7cfca8b
    - makedirs: True
    - mode: 744
    - require:
      - pip: python-etcd

calico-node:
  cmd.run:
    - name: /home/vagrant/calicoctl node --ip={{ grains.node_ip }}
    - env:
      - ETCD_AUTHORITY: "{{ grains.api_servers }}:6666"
    - require:
      - kmod: ip6_tables
      - kmod: xt_set
      - service: docker
      - file: calicoctl
      - container_bridge: cbr0

calico-ip-pool-reset:
  cmd.run:
    - name: /home/vagrant/calicoctl ipv4 pool remove 192.160.0.0/16
    - env:
      - ETCD_AUTHORITY: "{{ grains.api_servers }}:6666"
    - require:
      - service: docker
      - file: calicoctl
      - cmd: calico-node

calico-ip-pool:
  cmd.run:
    - name: /home/vagrant/calicoctl ipv4 pool add {{ grains['cbr-cidr'] }}
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

python-pip:
  pkg.installed

python-devel:
  pkg.installed:
    - require:
      - pkg: gcc
      - pkg: openssl-devel

libffi-devel:
  pkg.installed:
    - require:
      - pkg: gcc
      - pkg: openssl-devel

gcc:
  pkg.installed

openssl-devel:
  pkg.installed

python-etcd:
  pip.installed:
    - reload_modules: True
    - require:
      - pkg: python-pip
      - pkg: python-devel
      - pkg: libffi-devel

python-sh:
  pip.installed:
    - name: sh
    - reload_modules: True
    - require:
      - pkg: python-pip

{% endif %}
