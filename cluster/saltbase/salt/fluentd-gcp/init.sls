/etc/kubernetes/manifests/fluentd-gcp.manifest:
  file.managed:
    - source: salt://fluentd-gcp/fluentd-gcp.manifest
    - user: root
    - group: root
    - mode: 644
    - makedirs: true
    - dir_mode: 755
