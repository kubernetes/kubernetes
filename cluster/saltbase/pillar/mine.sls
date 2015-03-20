{% if grains.cloud is defined and grains.cloud == 'gce' -%}
# On GCE, there is no Salt mine. We run standalone.
{% else %}
# Allow everyone to see cached values of who sits at what IP
{% set networkInterfaceName = "eth0" %}
{% if grains.networkInterfaceName is defined %}
  {% set networkInterfaceName = grains.networkInterfaceName %}
{% endif %}
mine_functions:
  network.ip_addrs: [{{networkInterfaceName}}]
  grains.items: []
{% endif -%}
