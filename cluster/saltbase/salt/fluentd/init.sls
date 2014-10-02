/etc/kubernetes/manifests/fluentd.manifest:
  file.managed:
    - source: salt://fluentd/fluentd.manifest
    - user: root
    - group: root
    - mode: 644
    - makedirs: true
    - dir_mode: 755

