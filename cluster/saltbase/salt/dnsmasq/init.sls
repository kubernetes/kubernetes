/etc/kubernetes/manifests/dnsmasq.manifest:
  file.managed:
    - source: salt://dnsmasq/dnsmasq.manifest
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - makedirs: true
    - dir_mode: 755
    - require:
      - service: docker
      - service: kubelet
