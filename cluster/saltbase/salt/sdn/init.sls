{% if grains['os_family'] == 'RedHat' %}

openvswitch:
  pkg:
    - installed
  service.running:
    - enable: True

sdn:
  cmd.wait:
    - name: /vagrant/network_closure.sh
    - watch:
      - pkg: docker-io
      - pkg: openvswitch

{% endif %}
