{% if not pillar.get('is_systemd') %}

supervisor:
  pkg:
    - installed

monit:
  pkg:
    - purged

/etc/supervisor/conf.d/docker.conf:
  file:
    - managed
    - source: salt://supervisor/docker.conf
    - user: root
    - group: root
    - mode: 644
    - makedirs: True
    - require_in:
      - pkg: supervisor
    - require:
      - file: /usr/sbin/docker-checker.sh

/usr/sbin/docker-checker.sh:
  file:
    - managed
    - source: salt://supervisor/docker-checker.sh
    - user: root
    - group: root
    - mode: 755
    - makedirs: True

/etc/supervisor/conf.d/kubelet.conf:
  file:
    - managed
    - source: salt://supervisor/kubelet.conf
    - user: root
    - group: root
    - mode: 644
    - makedirs: True
    - require_in: 
      - pkg: supervisor
    - require: 
      - file: /usr/sbin/kubelet-checker.sh

/usr/sbin/kubelet-checker.sh:
  file:
    - managed
    - source: salt://supervisor/kubelet-checker.sh
    - template: jinja
    - user: root
    - group: root
    - mode: 755
    - makedirs: True

{% if grains['roles'][0] == 'kubernetes-master' -%}
/etc/supervisor/conf.d/kube-addons.conf:
  file:
    - managed
    - source: salt://supervisor/kube-addons.conf
    - user: root
    - group: root
    - mode: 644
    - makedirs: True
    - require_in: 
      - pkg: supervisor
    - require: 
      - file: /usr/sbin/kube-addons-checker.sh

/usr/sbin/kube-addons-checker.sh:
  file:
    - managed
    - source: salt://supervisor/kube-addons-checker.sh
    - user: root
    - group: root
    - mode: 755
    - makedirs: True
{% endif %}

/etc/supervisor/supervisor_watcher.sh:
  file.managed:
    - source: salt://supervisor/supervisor_watcher.sh
    - user: root
    - group: root
    - mode: 755
    - makedirs: True

crontab -l | { cat; echo "* * * * * /etc/supervisor/supervisor_watcher.sh 2>&1 | logger"; } | crontab -:
  cmd.run:
  - unless: crontab -l | grep "* * * * * /etc/supervisor/supervisor_watcher.sh 2>&1 | logger"

supervisor-service:
  service:
    - running
    - name: supervisor
    - watch:
      - pkg: supervisor
      - file: /etc/supervisor/conf.d/*
    - require:
      - pkg: supervisor

{% endif %}
