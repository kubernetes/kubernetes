opencontrail-networking-minion:
  cmd.script:
    - unless: test -f /var/log/contrail/provision_minion.log
    - env:
      - 'OPENCONTRAIL_TAG': '{{ pillar.get('opencontrail_tag') }}'
      - 'OPENCONTRAIL_KUBERNETES_TAG': '{{ pillar.get('opencontrail_kubernetes_tag') }}'
    - source: https://raw.githubusercontent.com/juniper/contrail-kubernetes/{{ pillar.get('opencontrail_kubernetes_tag') }}/cluster/provision_minion.sh
    - source_hash: https://raw.githubusercontent.com/juniper/contrail-kubernetes/{{ pillar.get('opencontrail_kubernetes_tag') }}/cluster/manifests.hash
    - cwd: /
    - user: root
    - group: root
    - mode: 755
    - shell: /bin/bash
