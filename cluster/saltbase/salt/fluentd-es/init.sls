/etc/kubernetes/manifests/fluentd-es.manifest:
  file.managed:
    - source: salt://fluentd-es/fluentd-es.manifest
    - user: root
    - group: root
    - mode: 644
    - makedirs: true
    - dir_mode: 755
