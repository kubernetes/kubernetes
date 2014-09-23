{% if grains['os_family'] == 'RedHat' %}
{% set environment_file = '/etc/sysconfig/scheduler' %}
{% else %}
{% set environment_file = '/etc/default/scheduler' %}
{% endif %}

{{ environment_file }}:
  file.managed:
    - source: salt://scheduler/default
    - template: jinja
    - user: root
    - group: root
    - mode: 644

/usr/local/bin/scheduler:
  file.managed:
    - source: salt://kube-bins/scheduler
    - user: root
    - group: root
    - mode: 755

{% if grains['os_family'] == 'RedHat' %}

/usr/lib/systemd/system/scheduler.service:
  file.managed:
    - source: salt://scheduler/scheduler.service
    - user: root
    - group: root

{% else %}

/etc/init.d/scheduler:
  file.managed:
    - source: salt://scheduler/initd
    - user: root
    - group: root
    - mode: 755

{% endif %}

scheduler:
  group.present:
    - system: True
  user.present:
    - system: True
    - gid_from_name: True
    - shell: /sbin/nologin
    - home: /var/scheduler
    - require:
      - group: scheduler
  service.running:
    - enable: True
    - watch:
      - file: /usr/local/bin/scheduler
      - file: {{ environment_file }}
{% if grains['os_family'] != 'RedHat' %}
      - file: /etc/init.d/scheduler
{% endif %}


