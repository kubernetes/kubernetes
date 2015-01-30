{% if grains['os_family'] != 'RedHat' %}

monit:
  pkg:
    - installed

/etc/monit/conf.d/etcd:
  file:
    - managed
    - source: salt://monit/etcd
    - user: root
    - group: root
    - mode: 644

monit-service:
  service:
    - running
    - name: monit 
    - watch:
      - pkg: monit 
      - file: /etc/monit/conf.d/etcd

{% endif %}