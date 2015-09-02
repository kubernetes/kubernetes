opencontrail-networking-minion:
  cmd.script:
    - source: https://raw.githubusercontent.com/juniper/contrail-kubernetes/vrouter-manifest/cluster/provision_minion.sh
    - source_hash: https://raw.githubusercontent.com/juniper/contrail-kubernetes/vrouter-manifest/cluster/manifests.hash
    - cwd: /
    - user: root
    - group: root
    - mode: 755
    - shell: /bin/bash
