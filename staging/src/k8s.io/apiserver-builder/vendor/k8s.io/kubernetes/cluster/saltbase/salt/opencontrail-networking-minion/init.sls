opencontrail-networking-minion:
  cmd.script:
    - unless: test -f /var/log/contrail/provision_minion.log
    - env:
      - 'OPENCONTRAIL_TAG': '{{ pillar.get('opencontrail_tag') }}'
      - 'OPENCONTRAIL_KUBERNETES_TAG': '{{ pillar.get('opencontrail_kubernetes_tag') }}'
      - 'OPENCONTRAIL_PUBLIC_SUBNET': '{{ pillar.get('opencontrail_public_subnet') }}'
      - 'SERVICE_CLUSTER_IP_RANGE': '{{ pillar.get('service_cluster_ip_range') }}'
    - source: https://raw.githubusercontent.com/juniper/contrail-kubernetes/{{ pillar.get('opencontrail_kubernetes_tag') }}/cluster/provision_minion.sh
    - source_hash: https://raw.githubusercontent.com/juniper/contrail-kubernetes/{{ pillar.get('opencontrail_kubernetes_tag') }}/cluster/manifests.hash
    - cwd: /
    - user: root
    - group: root
    - mode: 755
    - shell: /bin/bash
