{% if not pillar.get('is_systemd') %}

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

{% if "kubernetes-pool" in grains.get('roles', []) %}
/etc/monit/conf.d/kube-proxy:
  file:
    - managed
    - source: salt://monit/kube-proxy
    - user: root
    - group: root
    - mode: 644
{% endif %}

{% if grains['roles'][0] == 'kubernetes-master' -%}
/etc/monit/conf.d/kube-addons:
  file:
    - managed
    - source: salt://monit/kube-addons
    - user: root
    - group: root
    - mode: 644
{% endif %}

/etc/monit/monit_watcher.sh:
  file.managed:
    - source: salt://monit/monit_watcher.sh
    - user: root
    - group: root
    - mode: 755

crontab -l | { cat; echo "* * * * * /etc/monit/monit_watcher.sh 2>&1 | logger"; } | crontab -:
  cmd.run:
  - unless: crontab -l | grep "* * * * * /etc/monit/monit_watcher.sh 2>&1 | logger"

monit-service:
  service:
    - running
    - name: monit
    - watch:
      - pkg: monit
      - file: /etc/monit/conf.d/*

{% endif %}
