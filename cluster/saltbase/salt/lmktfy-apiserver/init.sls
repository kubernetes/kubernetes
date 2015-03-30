{% if grains['os_family'] == 'RedHat' %}
{% set environment_file = '/etc/sysconfig/lmktfy-apiserver' %}
{% else %}
{% set environment_file = '/etc/default/lmktfy-apiserver' %}
{% endif %}

{{ environment_file }}:
  file.managed:
    - source: salt://lmktfy-apiserver/default
    - template: jinja
    - user: root
    - group: root
    - mode: 644

/usr/local/bin/lmktfy-apiserver:
  file.managed:
    - source: salt://lmktfy-bins/lmktfy-apiserver
    - user: root
    - group: root
    - mode: 755

{% if grains['os_family'] == 'RedHat' %}

/usr/lib/systemd/system/lmktfy-apiserver.service:
  file.managed:
    - source: salt://lmktfy-apiserver/lmktfy-apiserver.service
    - user: root
    - group: root

{% else %}

/etc/init.d/lmktfy-apiserver:
  file.managed:
    - source: salt://lmktfy-apiserver/initd
    - user: root
    - group: root
    - mode: 755

{% endif %}

{% if grains.cloud is defined %}
{% if grains.cloud in ['aws', 'gce', 'vagrant'] %}
# TODO: generate and distribute tokens on other cloud providers.
/srv/lmktfyrnetes/known_tokens.csv:
  file.managed:
    - source: salt://lmktfy-apiserver/known_tokens.csv
    - user: lmktfy-apiserver
    - group: lmktfy-apiserver
    - mode: 400
    - watch:
      - user: lmktfy-apiserver
      - group: lmktfy-apiserver
    - watch_in:
      - service: lmktfy-apiserver
{% endif %}
{% endif %}

lmktfy-apiserver:
  group.present:
    - system: True
  user.present:
    - system: True
    - gid_from_name: True
    - groups:
      - lmktfy-cert
    - shell: /sbin/nologin
    - home: /var/lmktfy-apiserver
    - require:
      - group: lmktfy-apiserver
  service.running:
    - enable: True
    - watch:
      - file: {{ environment_file }}
      - file: /usr/local/bin/lmktfy-apiserver
{% if grains['os_family'] != 'RedHat' %}
      - file: /etc/init.d/lmktfy-apiserver
{% endif %}
