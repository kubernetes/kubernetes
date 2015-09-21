{% if grains['os_family'] == 'RedHat' %}
{% set environment_file = '/etc/sysconfig/kube-scheduler' %}
{% else %}
{% set environment_file = '/etc/default/kube-scheduler' %}
{% endif %}

{{ environment_file }}:
  file.managed:
    - source: salt://kube-scheduler/default
    - template: jinja
    - user: root
    - group: root
    - mode: 644

/usr/local/bin/kube-scheduler:
  file.managed:
    - source: salt://kube-bins/kube-scheduler
    - user: root
    - group: root
    - mode: 755

{% if grains['os_family'] == 'RedHat' %}

/usr/lib/systemd/system/kube-scheduler.service:
  file.managed:
    - source: salt://kube-scheduler/kube-scheduler.service
    - user: root
    - group: root

{% else %}

/etc/init.d/kube-scheduler:
  file.managed:
    - source: salt://kube-scheduler/initd
    - user: root
    - group: root
    - mode: 755

{% endif %}

kube-scheduler:
  group.present:
    - system: True
  user.present:
    - system: True
    - gid_from_name: True
    - shell: /sbin/nologin
    - home: /var/kube-scheduler
    - require:
      - group: kube-scheduler
  service.running:
    - enable: True
    - watch:
      - file: /usr/local/bin/kube-scheduler
      - file: {{ environment_file }}
{% if grains['os_family'] != 'RedHat' %}
      - file: /etc/init.d/kube-scheduler
{% endif %}


