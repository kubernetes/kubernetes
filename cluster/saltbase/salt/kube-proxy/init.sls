/var/lib/kube-proxy/kubeconfig:
  file.managed:
    - source: salt://kube-proxy/kubeconfig
    - user: root
    - group: root
    - mode: 400
    - makedirs: true

# kube-proxy in a static pod
{% if pillar.get('kube_proxy_daemonset', '').lower() != 'true' %}
/etc/kubernetes/manifests/kube-proxy.manifest:
  file.managed:
    - source: salt://kube-proxy/kube-proxy.manifest
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - makedirs: true
    - dir_mode: 755
    - context:
        # Increasing to 100m to avoid CPU starvation on full nodes.
        # Any change here should be accompanied by a proportional change in CPU
        # requests of other per-node add-ons (e.g. fluentd).
        cpurequest: '100m'
    - require:
      - service: docker
      - service: kubelet
{% endif %}

/var/log/kube-proxy.log:
  file.managed:
    - user: root
    - group: root
    - mode: 644

#stop legacy kube-proxy service
stop_kube-proxy:
  service.dead:
    - name: kube-proxy
    - enable: None
