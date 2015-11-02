opencontrail-kubelet-plugin:
  cmd.script:
    - unless: test -f /etc/contrail/kubelet_plugin_install.ok
    - env:
      - 'OPENCONTRAIL_KUBERNETES_TAG': '{{ pillar.get('opencontrail_kubernetes_tag') }}'
    - source: https://raw.githubusercontent.com/juniper/contrail-kubernetes/{{ pillar.get('opencontrail_kubernetes_tag') }}/cluster/provision_kubelet_plugin.sh
    - source_hash: https://raw.githubusercontent.com/juniper/contrail-kubernetes/{{ pillar.get('opencontrail_kubernetes_tag') }}/cluster/manifests.hash
    - cwd: /
    - user: root
    - group: root
    - mode: 755
    - shell: /bin/bash
