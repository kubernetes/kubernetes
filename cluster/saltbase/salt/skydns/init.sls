/etc/kubernetes/manifests/skydns.manifest:
  file.managed:
    - source: salt://skydns/skydns.manifest
    - user: root
    - group: root
    - mode: 644
    - makedirs: true
    - dir_mode: 755
