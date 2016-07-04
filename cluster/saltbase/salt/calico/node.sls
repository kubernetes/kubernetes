{% if pillar.get('network_policy_provider', '').lower() == 'calico' %}

calicoctl:
  file.managed:
    - name: /usr/bin/calicoctl
    - source: https://github.com/projectcalico/calico-docker/releases/download/v0.19.0/calicoctl
    - source_hash: sha256=6db00c94619e82d878d348c4e1791f8d2f0db59075f6c8e430fefae297c54d96
    - makedirs: True
    - mode: 744

calico-node:
  cmd.run:
    - name: calicoctl node
    - unless: docker ps | grep calico-node
    - env:
      - ETCD_AUTHORITY: "{{ grains.api_servers }}:6666"
      - CALICO_NETWORKING: "false"
    - require:
      - kmod: ip6_tables
      - kmod: xt_set
      - service: docker
      - file: calicoctl

calico-cni:
  file.managed:
    - name: /opt/cni/bin/calico
    - source: https://github.com/projectcalico/calico-cni/releases/download/v1.3.1/calico 
    - source_hash: sha256=ac05cb9254b5aaa5822cf10325983431bd25489147f2edf9dec7e43d99c43e77
    - makedirs: True
    - mode: 744

calico-cni-config:
  file.managed:
    - name: /etc/cni/net.d/10-calico.conf
    - source: salt://calico/10-calico.conf
    - makedirs: True
    - mode: 644
    - template: jinja

ip6_tables:
  kmod.present

xt_set:
  kmod.present

{% endif -%}
