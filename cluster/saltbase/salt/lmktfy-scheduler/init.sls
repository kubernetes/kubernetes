{% if grains['os_family'] == 'RedHat' %}
{% set environment_file = '/etc/sysconfig/lmktfy-scheduler' %}
{% else %}
{% set environment_file = '/etc/default/lmktfy-scheduler' %}
{% endif %}

{{ environment_file }}:
  file.managed:
    - source: salt://lmktfy-scheduler/default
    - template: jinja
    - user: root
    - group: root
    - mode: 644

/usr/local/bin/lmktfy-scheduler:
  file.managed:
    - source: salt://lmktfy-bins/lmktfy-scheduler
    - user: root
    - group: root
    - mode: 755

{% if grains['os_family'] == 'RedHat' %}

/usr/lib/systemd/system/lmktfy-scheduler.service:
  file.managed:
    - source: salt://lmktfy-scheduler/lmktfy-scheduler.service
    - user: root
    - group: root

{% else %}

/etc/init.d/lmktfy-scheduler:
  file.managed:
    - source: salt://lmktfy-scheduler/initd
    - user: root
    - group: root
    - mode: 755

{% endif %}

lmktfy-scheduler:
  group.present:
    - system: True
  user.present:
    - system: True
    - gid_from_name: True
    - shell: /sbin/nologin
    - home: /var/lmktfy-scheduler
    - require:
      - group: lmktfy-scheduler
  service.running:
    - enable: True
    - watch:
      - file: /usr/local/bin/lmktfy-scheduler
      - file: {{ environment_file }}
{% if grains['os_family'] != 'RedHat' %}
      - file: /etc/init.d/lmktfy-scheduler
{% endif %}


