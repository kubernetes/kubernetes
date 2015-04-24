/etc/kubernetes/manifests/kube-controller-manager.manifest:
  file.managed:
    - source: salt://kube-controller-manager/kube-controller-manager.manifest
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - makedirs: true
    - dir_mode: 755

/var/log/kube-controller-manager.log:
  file.managed:
    - user: root
    - group: root
    - mode: 644

stop-legacy-kube_controller_manager:
  service.dead:
    - name: kube-controller-manager
    - enable: None

