{% if pillar.get('network_policy_provider', '').lower() == 'calico' %}

ip6_tables:
  kmod.present

xt_set:
  kmod.present

{% endif -%}
