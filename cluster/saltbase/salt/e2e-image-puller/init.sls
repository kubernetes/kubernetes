/etc/kubernetes/manifests/e2e-image-puller.manifest:
  file.managed:
    - source: salt://e2e-image-puller/e2e-image-puller.manifest
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - makedirs: true
    - dir_mode: 755
    - require:
      - service: docker
      - service: kubelet
