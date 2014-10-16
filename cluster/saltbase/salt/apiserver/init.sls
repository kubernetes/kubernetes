{% if grains['os_family'] == 'RedHat' %}
{% set environment_file = '/etc/sysconfig/apiserver' %}
{% else %}
{% set environment_file = '/etc/default/apiserver' %}
{% endif %}

{{ environment_file }}:
  file.managed:
    - source: salt://apiserver/default
    - template: jinja
    - user: root
    - group: root
    - mode: 644

/usr/local/bin/apiserver:
  file.managed:
    - source: salt://kube-bins/apiserver
    - user: root
    - group: root
    - mode: 755

{% if grains['os_family'] == 'RedHat' %}

/usr/lib/systemd/system/apiserver.service:
  file.managed:
    - source: salt://apiserver/apiserver.service
    - user: root
    - group: root

{% else %}

/etc/init.d/apiserver:
  file.managed:
    - source: salt://apiserver/initd
    - user: root
    - group: root
    - mode: 755

{% endif %}

apiserver:
  group.present:
    - system: True
  user.present:
    - system: True
    - gid_from_name: True
    - shell: /sbin/nologin
    - home: /var/apiserver
    - require:
      - group: apiserver
  service.running:
    - enable: True
    - watch:
      - file: {{ environment_file }}
      - file: /usr/local/bin/apiserver
{% if grains['os_family'] != 'RedHat' %}
      - file: /etc/init.d/apiserver
{% endif %}
