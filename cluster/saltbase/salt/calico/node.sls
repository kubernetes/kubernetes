{% if pillar.get('network_provider', '').lower() == 'calico' %}

include:
  - docker

calicoctl:
  file.managed:
    - name: /usr/bin/calicoctl
    - source: https://github.com/projectcalico/calico-docker/releases/download/v0.10.0/calicoctl
    - source_hash: sha512=5dd8110cebfc00622d49adddcccda9d4906e6bca8a777297e6c0ffbcf0f7e40b42b0d6955f2e04b457b0919cb2d5ce39d2a3255d34e6ba36e8350f50319b3896
    - makedirs: True
    - mode: 744

calico-plugin:
  file.managed:
    - name: /usr/libexec/kubernetes/kubelet-plugins/net/exec/calico/calico
    - source: https://github.com/projectcalico/calico-kubernetes/releases/download/v0.6.0/calico_kubernetes
    - source_hash: sha512=5b95b837071daf6e413c76f22d7d423e74be9c0525230cd074ff3f8ecb724e6f5a7db8a31d6473b5f783096a4f12ce688dde3b5258f0d5949947df3635ac1613
    - makedirs: True
    - mode: 744
    - require_in:
      - service: kubelet

plugin-config:
  file.managed:
    - name: /usr/libexec/kubernetes/kubelet-plugins/net/exec/calico/calico_kubernetes.ini
    - source: salt://calico/calico_kubernetes.ini
    - template: jinja
    - makedirs: True
    - mode: 744

calico-node:
  cmd.run:
    - name: calicoctl node
    - unless: docker ps | grep calico-node
    - env:
      - ETCD_AUTHORITY: "{{ grains.api_servers }}:6666"
    - require:
      - kmod: ip6_tables
      - kmod: xt_set
      - service: docker
      - file: calicoctl
      - file: plugin-config
      - file: calico-plugin

calico-ip-pool-reset:
  cmd.run:
    - name: calicoctl pool remove 192.168.0.0/16
    - onlyif: calicoctl pool show | grep 192.168.0.0/16
    - env:
      - ETCD_AUTHORITY: "{{ grains.api_servers }}:6666"
    - require:
      - service: docker
      - file: calicoctl
      - cmd: calico-node
    - require_in:
      - service: kubelet

calico-ip-pool:
  cmd.run:
    - name: calicoctl pool add {{ grains['cbr-cidr'] }}
    - unless: calicoctl pool show | grep {{ grains['cbr-cidr'] }}
    - env:
      - ETCD_AUTHORITY: "{{ grains.api_servers }}:6666"
    - require:
      - service: docker
      - file: calicoctl
      - cmd: calico-node
    - require_in:
      - service: kubelet

ip6_tables:
  kmod.present

xt_set:
  kmod.present

{% endif %}
