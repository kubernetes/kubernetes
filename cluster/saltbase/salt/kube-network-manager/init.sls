/etc/kubernetes/manifests/kube-network-manager.manifest:
  file.managed:
    - source: salt://kube-network-manager/kube-network-manager.manifest
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - makedirs: true
    - dir_mode: 755
    - require:
      - service: docker
      - service: kubelet

/var/log/kube-network-manager.log:
  file.managed:
    - user: root
    - group: root
    - mode: 644

stop_kube-network-manager:
  service.dead:
    - name: kube-network-manager
    - enable: None
