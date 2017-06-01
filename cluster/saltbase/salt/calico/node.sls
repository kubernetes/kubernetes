{% if pillar.get('network_policy_provider', '').lower() == 'calico' %}

calico-node:
  file.managed:
    - name: /etc/kubernetes/manifests/calico-node.manifest
    - source: salt://calico/calico-node.manifest
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - makedirs: true
    - dir_mode: 755
    - require:
      - kmod: ip6_tables
      - kmod: xt_set
      - service: docker
      - service: kubelet

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
