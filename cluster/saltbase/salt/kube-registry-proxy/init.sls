/etc/kubernetes/manifests/kube-registry-proxy.yaml:
  file.managed:
    - source: salt://kube-registry-proxy/kube-registry-proxy.yaml
    - user: root
    - group: root
    - mode: 644
    - makedirs: True
    - dir_mode: 755
