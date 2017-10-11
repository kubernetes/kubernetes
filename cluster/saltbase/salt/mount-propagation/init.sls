{% if not pillar.get('is_systemd') %}

/etc/init.d/mount-propagation:
  file.managed:
    - source: salt://mount-propagation/initd
    - user: root
    - group: root
    - mode: 755

mount-propagation:
  service.running:
    - enable: True
    - watch:
      - file: /etc/init.d/mount-propagation
{%- endif %}
