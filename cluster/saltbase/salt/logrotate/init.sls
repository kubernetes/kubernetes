logrotate:
  pkg:
    - installed

{% set logrotate_files = ['scheduler', 'kube-proxy', 'kubelet', 'apiserver', 'controller-manager'] %}
{% for file in logrotate_files %}
/etc/logrotate.d/{{ file }}:
  file:
    - managed
    - source: salt://logrotate/conf
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - context:
      file: {{ file }}
{% endfor %}

/etc/logrotate.d/docker:
  file:
    - managed
    - source: salt://logrotate/docker
    - template: jinja
    - user: root
    - group: root
    - mode: 644

/etc/cron.hourly/logrotate:
  file:
    - managed
    - source: salt://logrotate/cron
    - template: jinja
    - user: root
    - group: root
    - mode: 755
