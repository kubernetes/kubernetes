{% if grains.network_mode is defined and grains.network_mode == 'openvswitch' %}


sdn:
  cmd.script:
    - source: /kubernetes-vagrant/network_closure.sh
    - require:
      - pkg: docker-io
    - cwd: /
    - user: root
    - group: root
    - shell: /bin/bash
{% endif %}
