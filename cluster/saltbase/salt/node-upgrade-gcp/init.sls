{% if grains['cloud'] is defined and grains['cloud'] == 'gce' %}
/etc/kubernetes/node-upgrade-gcp.sh:
  file.managed:
    - source: salt://node-upgrade-gcp/node-upgrade-gcp.sh
    - user: root
    - group: root
    - mode: 755
  {% if grains['os_family'] == 'RedHat' %}
/usr/lib/systemd/system/node-upgrade-gcp.service:
  file.managed:
    - source: salt://node-upgrade-gcp/node-upgrade-gcp.service
    - user: root
    - group: root
  {% else %}
/etc/init.d/node-upgrade-gcp:
  file.managed:
    - source: salt://node-upgrade-gcp/initd
    - user: root
    - group: root
    - mode: 755
  {% endif %}
{% endif %}

node-upgrade-gcp:
  service.running
    - enable: True
