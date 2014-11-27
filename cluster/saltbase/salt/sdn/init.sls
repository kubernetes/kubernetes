{% if grains.network_mode is defined and grains.network_mode == 'openvswitch' %}

openvswitch:
  pkg:
    - installed
  service.running:
    - enable: True

sdn:
  cmd.wait:
    - name: /kubernetes-vagrant/network_closure.sh
    - watch:
      - pkg: docker-io
      - pkg: openvswitch
{% endif %}
