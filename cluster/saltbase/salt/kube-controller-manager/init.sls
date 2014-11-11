{% if grains['os_family'] == 'RedHat' %}
{% set environment_file = '/etc/sysconfig/kube-controller-manager' %}
{% else %}
{% set environment_file = '/etc/default/kube-controller-manager' %}
{% endif %}

{{ environment_file }}:
  file.managed:
    - source: salt://kube-controller-manager/default
    - template: jinja
    - user: root
    - group: root
    - mode: 644

/usr/local/bin/kube-controller-manager:
  file.managed:
    - source: salt://kube-bins/kube-controller-manager
    - user: root
    - group: root
    - mode: 755

{% if grains['os_family'] == 'RedHat' %}

/usr/lib/systemd/system/kube-controller-manager.service:
  file.managed:
    - source: salt://kube-controller-manager/kube-controller-manager.service
    - user: root
    - group: root

{% else %}

/etc/init.d/kube-controller-manager:
  file.managed:
    - source: salt://kube-controller-manager/initd
    - user: root
    - group: root
    - mode: 755

{% endif %}

kube-controller-manager:
  group.present:
    - system: True
  user.present:
    - system: True
    - gid_from_name: True
    - shell: /sbin/nologin
    - home: /var/kube-controller-manager
    - require:
      - group: kube-controller-manager
  service.running:
    - enable: True
    - watch:
      - file: /usr/local/bin/kube-controller-manager
      - file: {{ environment_file }}
{% if grains['os_family'] != 'RedHat' %}
      - file: /etc/init.d/kube-controller-manager
{% endif %}


