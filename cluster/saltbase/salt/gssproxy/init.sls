/etc/gssproxy/gssproxy.conf:
  file.managed:
    - source: salt://gssproxy/gssproxy.conf
    - user: root
    - group: root

gssproxy:
  pkg:
    - installed
  service.running:
    - enable: true
{% if grains['os_family'] != 'RedHat' %}
    - file: /etc/init.d/gssproxy
{% endif %}
    - watch:
      - file: /etc/gssproxy/gssproxy.conf
