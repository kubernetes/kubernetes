{% if grains['os_family'] != 'RedHat' %}

monit:
  pkg:
    - installed

{% if "lmktfyrnetes-master" in grains.get('roles', []) %}
/etc/monit/conf.d/etcd:
  file:
    - managed
    - source: salt://monit/etcd
    - user: root
    - group: root
    - mode: 644
{% endif %}

/etc/monit/conf.d/docker:
  file:
    - managed
    - source: salt://monit/docker
    - user: root
    - group: root
    - mode: 644

{% if "lmktfyrnetes-pool" in grains.get('roles', []) %}
/etc/monit/conf.d/lmktfylet:
  file:
    - managed
    - source: salt://monit/lmktfylet
    - user: root
    - group: root
    - mode: 644

/etc/monit/conf.d/lmktfy-proxy:
  file:
    - managed
    - source: salt://monit/lmktfy-proxy
    - user: root
    - group: root
    - mode: 644
{% endif %}

monit-service:
  service:
    - running
    - name: monit
    - watch:
      - pkg: monit
      - file: /etc/monit/conf.d/*

{% endif %}
