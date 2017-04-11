/etc/kubernetes/manifests/glbc.manifest:
  file.managed:
    - source: salt://l7-gcp/glbc.manifest
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - makedirs: true
    - dir_mode: 755

/var/log/glbc.log:
  file.managed:
    - user: root
    - group: root
    - mode: 644


