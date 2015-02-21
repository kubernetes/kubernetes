{% if grains.network_mode is defined and grains.network_mode == 'openvswitch' %}

sdn:
  cmd.wait:
    - name: /kubernetes-vagrant/network_closure.sh
    - watch:
      - sls: docker
{% endif %}
