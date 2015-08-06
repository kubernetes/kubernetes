# Copy kube-scheduler manifest to manifests folder for kubelet.
# The ordering of salt states for service docker, kubelet and
# master-addon below is very important to avoid the race between
# salt restart docker or kubelet and kubelet start master components.
# Please see http://issue.k8s.io/10122#issuecomment-114566063
# for detail explanation on this very issue.
/etc/kubernetes/manifests/kube-scheduler.manifest:
  file.managed:
    - source: salt://kube-scheduler/kube-scheduler.manifest
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - makedirs: true
    - dir_mode: 755
    - require:
      - service: docker
      - service: kubelet

/var/log/kube-scheduler.log:
  file.managed:
    - user: root
    - group: root
    - mode: 644

#stop legacy kube-scheduler service 
stop_kube-scheduler:
  service.dead:
    - name: kube-scheduler
    - enable: None
