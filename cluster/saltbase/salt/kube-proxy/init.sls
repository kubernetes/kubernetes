{% if grains['os_family'] == 'RedHat' %}
{% set environment_file = '/etc/sysconfig/kube-proxy' %}
{% else %}
{% set environment_file = '/etc/default/kube-proxy' %}
{% endif %}

/usr/local/bin/kube-proxy:
  file.managed:
    - source: salt://kube-bins/kube-proxy
    - user: root
    - group: root
    - mode: 755

{% if grains['os_family'] == 'RedHat' %}

/usr/lib/systemd/system/kube-proxy.service:
  file.managed:
    - source: salt://kube-proxy/kube-proxy.service
    - user: root
    - group: root

{% else %}

/etc/init.d/kube-proxy:
  file.managed:
    - source: salt://kube-proxy/initd
    - user: root
    - group: root
    - mode: 755

{% endif %}

{{ environment_file }}:
  file.managed:
    - source: salt://kube-proxy/default
    - template: jinja
    - user: root
    - group: root
    - mode: 644

kube-proxy:
  group.present:
    - system: True
  user.present:
    - system: True
    - gid_from_name: True
    - shell: /sbin/nologin
    - home: /var/kube-proxy
    - require:
      - group: kube-proxy
  service.running:
    - enable: True
    - watch:
      - file: {{ environment_file }}
{% if grains['os_family'] != 'RedHat' %}
      - file: /etc/init.d/kube-proxy
{% endif %}
      - file: /var/lib/kube-proxy/kubeconfig

/var/lib/kube-proxy/kubeconfig:
  file.managed:
    - source: salt://kube-proxy/kubeconfig
    - user: root
    - group: root
    - mode: 400
    - makedirs: true
