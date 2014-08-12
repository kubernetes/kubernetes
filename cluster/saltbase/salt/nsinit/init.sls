{% if grains['os_family'] != 'RedHat' %}
build-essential:
  pkg:
    - installed
{% endif %}

nsinit:
  cmd.script:
    - user: root
    - shell: /bin/bash
    - source: salt://nsinit/install.sh
