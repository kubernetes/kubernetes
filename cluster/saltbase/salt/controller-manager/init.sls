{% if grains['os_family'] == 'RedHat' %}
{% set environment_file = '/etc/sysconfig/controller-manager' %}
{% else %}
{% set environment_file = '/etc/default/controller-manager' %}
{% endif %}

{{ environment_file }}:
  file.managed:
    - source: salt://controller-manager/default
    - template: jinja
    - user: root
    - group: root
    - mode: 644

/usr/local/bin/controller-manager:
  file.managed:
    - source: salt://kube-bins/controller-manager
    - user: root
    - group: root
    - mode: 755

{% if grains['os_family'] == 'RedHat' %}

/usr/lib/systemd/system/controller-manager.service:
  file.managed:
    - source: salt://controller-manager/controller-manager.service
    - user: root
    - group: root

{% else %}

/etc/init.d/controller-manager:
  file.managed:
    - source: salt://controller-manager/initd
    - user: root
    - group: root
    - mode: 755

{% endif %}

controller-manager:
  group.present:
    - system: True
  user.present:
    - system: True
    - gid_from_name: True
    - shell: /sbin/nologin
    - home: /var/controller-manager
    - require:
      - group: controller-manager
  service.running:
    - enable: True
    - watch:
      - file: /usr/local/bin/controller-manager
      - file: {{ environment_file }}
{% if grains['os_family'] != 'RedHat' %}
      - file: /etc/init.d/controller-manager
{% endif %}


