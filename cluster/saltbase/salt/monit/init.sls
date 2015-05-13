{% if grains['os_family'] != 'RedHat' %}

monit:
  pkg:
    - installed

/etc/monit/conf.d/docker:
  file:
    - managed
    - source: salt://monit/docker
    - user: root
    - group: root
    - mode: 644

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

monit-service:
  service:
    - running
    - name: monit
    - watch:
      - pkg: monit
      - file: /etc/monit/conf.d/*

{% endif %}
