opencontrail-vrouter-kernel:
  cmd.script:
    - unless: test -f /etc/contrail/vrouter_kmod.ok
    - env:
      - 'OPENCONTRAIL_TAG': '{{ pillar.get('opencontrail_tag') }}'
    - source: https://raw.githubusercontent.com/juniper/contrail-kubernetes/{{ pillar.get('opencontrail_kubernetes_tag') }}/cluster/provision_vrouter_kernel.sh
    - source_hash: https://raw.githubusercontent.com/juniper/contrail-kubernetes/{{ pillar.get('opencontrail_kubernetes_tag') }}/cluster/manifests.hash
    - cwd: /
    - user: root
    - group: root
    - mode: 755
    - shell: /bin/bash
