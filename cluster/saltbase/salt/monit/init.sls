{% if grains['os_family'] != 'RedHat' %}

monit:
  pkg:
    - installed

{% if "kubernetes-master" in grains.get('roles', []) %}
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

{% if "kubernetes-pool" in grains.get('roles', []) %}
/etc/monit/conf.d/kubelet:
  file:
    - managed
    - source: salt://monit/kubelet
    - user: root
    - group: root
    - mode: 644

/etc/monit/conf.d/kube-proxy:
  file:
    - managed
    - source: salt://monit/kube-proxy
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
