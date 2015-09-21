/etc/kubernetes/manifests/cadvisor.manifest:
  file.managed:
    - source: salt://cadvisor/cadvisor.manifest
    - user: root
    - group: root
    - mode: 644
    - makedirs: true
    - dir_mode: 755
