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
    - source_hash: sha512=a1a887ac06b6050bfd170c36b7af27e244d19ee64747224d99a167f6d48dc44b36e41c8c801a2d70301988157db52df5b1d90921677b747d6414b280f17eee75
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
