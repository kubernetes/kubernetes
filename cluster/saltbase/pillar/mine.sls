# On GCE, there is no Salt mine. We run standalone.
{% if grains.cloud != 'gce' -%}

# Allow everyone to see cached values of who sits at what IP
{% set networkInterfaceName = "eth0" %}
{% if grains.networkInterfaceName is defined %}
  {% set networkInterfaceName = grains.networkInterfaceName %}
{% endif %}
mine_functions:
  network.ip_addrs: [{{networkInterfaceName}}]
  grains.items: []

{% endif -%}
