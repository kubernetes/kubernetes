{% if grains.network_mode is defined and grains.network_mode == 'calico' %}

include:
  - docker

calicoctl:
  file.managed:
    - name: /home/vagrant/calicoctl
    - source: https://github.com/Metaswitch/calico-docker/releases/download/v0.4.2/calicoctl
    - source_hash: sha512=4fe30422bbf47fdade3ef7ba8927643cff8af578a81b4e6a101525229a583e18afbc3f7d5dee4d60b789139173a34fe15cb8c2c120014230ebd431b655a5fb1b
    - makedirs: True
    - mode: 744

calico-network-plugin:
  file.managed:
    - name: /usr/libexec/kubernetes/kubelet-plugins/net/exec/calico/calico
    - source: https://github.com/Metaswitch/calico-docker/releases/download/v0.4.2/calico_kubernetes
    - source_hash: sha512=eb3a7b35668b4d8d180cc10f9937ed44752f7185907ef0559e0ea52eb55e837b0dca3a1a6ad75d52aa1481d2f63e3c5a09314222fa72e5d5c43c004d7283bac8
    - makedirs: True
    - mode: 744

calico-node:
  cmd.run:
    - name: /home/vagrant/calicoctl node --ip={{ grains.node_ip }} --node-image=calico/node:v0.4.2
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
