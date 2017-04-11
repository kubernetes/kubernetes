/opt/kubernetes/helpers:
  file.directory:
    - user: root
    - group: root
    - makedirs: True
    - dir_mode: 755

{% if pillar.get('is_systemd') %}
/opt/kubernetes/helpers/services:
  file.managed:
    - source: salt://salt-helpers/services
    - user: root
    - group: root
    - mode: 755
{% endif %}

{% if grains.get('os_family', '') == 'Debian' -%}
/opt/kubernetes/helpers/pkg:
  file.managed:
    - source: salt://salt-helpers/pkg-apt
    - user: root
    - group: root
    - mode: 755
{% endif %}
