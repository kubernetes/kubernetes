/etc/kubernetes/manifests/fluentd-es.json:
  file.managed:
    - source: salt://fluentd-es/fluentd-es.json
    - user: root
    - group: root
    - mode: 644
    - makedirs: true
    - dir_mode: 755
