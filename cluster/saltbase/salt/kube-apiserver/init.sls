{% if grains['os_family'] == 'RedHat' %}
{% set environment_file = '/etc/sysconfig/kube-apiserver' %}
{% else %}
{% set environment_file = '/etc/default/kube-apiserver' %}
{% endif %}

{{ environment_file }}:
  file.managed:
    - source: salt://kube-apiserver/default
    - template: jinja
    - user: root
    - group: root
    - mode: 644

/usr/local/bin/kube-apiserver:
  file.managed:
    - source: salt://kube-bins/kube-apiserver
    - user: root
    - group: root
    - mode: 755

{% if grains['os_family'] == 'RedHat' %}

/usr/lib/systemd/system/kube-apiserver.service:
  file.managed:
    - source: salt://kube-apiserver/kube-apiserver.service
    - user: root
    - group: root

{% else %}

/etc/init.d/kube-apiserver:
  file.managed:
    - source: salt://kube-apiserver/initd
    - user: root
    - group: root
    - mode: 755

{% endif %}

{% if grains.cloud is defined %}
{% if grains.cloud in ['aws', 'gce', 'vagrant'] %}
# TODO: generate and distribute tokens on other cloud providers.
/srv/kubernetes/known_tokens.csv:
  file.managed:
    - source: salt://kube-apiserver/known_tokens.csv
    - user: kube-apiserver
    - group: kube-apiserver
    - mode: 400
    - watch:
      - user: kube-apiserver
      - group: kube-apiserver
    - watch_in:
      - service: kube-apiserver
{% endif %}
{% endif %}

kube-apiserver:
  group.present:
    - system: True
  user.present:
    - system: True
    - gid_from_name: True
    - groups:
      - kube-cert
    - shell: /sbin/nologin
    - home: /var/kube-apiserver
    - require:
      - group: kube-apiserver
  service.running:
    - enable: True
    - watch:
      - file: {{ environment_file }}
      - file: /usr/local/bin/kube-apiserver
{% if grains['os_family'] != 'RedHat' %}
      - file: /etc/init.d/kube-apiserver
{% endif %}
