{% if grains['os_family'] == 'RedHat' %}
{% set environment_file = '/etc/sysconfig/lmktfy-proxy' %}
{% else %}
{% set environment_file = '/etc/default/lmktfy-proxy' %}
{% endif %}

/usr/local/bin/lmktfy-proxy:
  file.managed:
    - source: salt://lmktfy-bins/lmktfy-proxy
    - user: root
    - group: root
    - mode: 755

{% if grains['os_family'] == 'RedHat' %}

/usr/lib/systemd/system/lmktfy-proxy.service:
  file.managed:
    - source: salt://lmktfy-proxy/lmktfy-proxy.service
    - user: root
    - group: root

{% else %}

/etc/init.d/lmktfy-proxy:
  file.managed:
    - source: salt://lmktfy-proxy/initd
    - user: root
    - group: root
    - mode: 755

{% endif %}

{{ environment_file }}:
  file.managed:
    - source: salt://lmktfy-proxy/default
    - template: jinja
    - user: root
    - group: root
    - mode: 644

lmktfy-proxy:
  group.present:
    - system: True
  user.present:
    - system: True
    - gid_from_name: True
    - shell: /sbin/nologin
    - home: /var/lmktfy-proxy
    - require:
      - group: lmktfy-proxy
  service.running:
    - enable: True
    - watch:
      - file: {{ environment_file }}
{% if grains['os_family'] != 'RedHat' %}
      - file: /etc/init.d/lmktfy-proxy
{% endif %}
