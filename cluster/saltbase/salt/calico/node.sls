{% if pillar.get('network_policy_provider', '').lower() == 'calico' %}

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
