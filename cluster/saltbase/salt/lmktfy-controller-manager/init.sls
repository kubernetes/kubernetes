{% if grains['os_family'] == 'RedHat' %}
{% set environment_file = '/etc/sysconfig/lmktfy-controller-manager' %}
{% else %}
{% set environment_file = '/etc/default/lmktfy-controller-manager' %}
{% endif %}

{{ environment_file }}:
  file.managed:
    - source: salt://lmktfy-controller-manager/default
    - template: jinja
    - user: root
    - group: root
    - mode: 644

/usr/local/bin/lmktfy-controller-manager:
  file.managed:
    - source: salt://lmktfy-bins/lmktfy-controller-manager
    - user: root
    - group: root
    - mode: 755

{% if grains['os_family'] == 'RedHat' %}

/usr/lib/systemd/system/lmktfy-controller-manager.service:
  file.managed:
    - source: salt://lmktfy-controller-manager/lmktfy-controller-manager.service
    - user: root
    - group: root

{% else %}

/etc/init.d/lmktfy-controller-manager:
  file.managed:
    - source: salt://lmktfy-controller-manager/initd
    - user: root
    - group: root
    - mode: 755

{% endif %}

lmktfy-controller-manager:
  group.present:
    - system: True
  user.present:
    - system: True
    - gid_from_name: True
    - shell: /sbin/nologin
    - home: /var/lmktfy-controller-manager
    - require:
      - group: lmktfy-controller-manager
  service.running:
    - enable: True
    - watch:
      - file: /usr/local/bin/lmktfy-controller-manager
      - file: {{ environment_file }}
{% if grains['os_family'] != 'RedHat' %}
      - file: /etc/init.d/lmktfy-controller-manager
{% endif %}


