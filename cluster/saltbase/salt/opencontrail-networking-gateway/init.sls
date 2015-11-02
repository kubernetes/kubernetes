opencontrail-networking-gateway:
  cmd.script:
    - unless: test -f /etc/contrail/vrouter_sgw_prov_run.ok
    - env:
      - 'OPENCONTRAIL_TAG': '{{ pillar.get('opencontrail_tag') }}'
      - 'OPENCONTRAIL_KUBERNETES_TAG': '{{ pillar.get('opencontrail_kubernetes_tag') }}'
      - 'OPENCONTRAIL_PUBLIC_SUBNET': '{{ pillar.get('opencontrail_public_subnet') }}'
      - 'NETWORK_PROVIDER_GATEWAY_ON_MINION': '{{ pillar.get('network_provider_gw_on_minion') }}'
    - source: https://raw.githubusercontent.com/juniper/contrail-kubernetes/{{ pillar.get('opencontrail_kubernetes_tag') }}/cluster/provision_gateway.sh
    - source_hash: https://raw.githubusercontent.com/juniper/contrail-kubernetes/{{ pillar.get('opencontrail_kubernetes_tag') }}/cluster/manifests.hash
    - cwd: /
    - user: root
    - group: root
    - mode: 755
    - shell: /bin/bash
