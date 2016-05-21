{% if pillar.get('network_policy_provider', '').lower() == 'calico' %}

calico-policy-controller:
  file.managed:
    - name: /etc/kubernetes/manifests/calico-policy-controller.manifest
    - source: salt://calico/calico-policy-controller.manifest
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - makedirs: true
    - dir_mode: 755
    - context:
        cpurequest: '20m'
    - require:
      - service: docker
      - service: kubelet

{% endif -%}
