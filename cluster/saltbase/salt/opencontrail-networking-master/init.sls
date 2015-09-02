opencontrail-networking-master:
  cmd.script:
    - source: https://raw.githubusercontent.com/rombie/contrail-kubernetes/manifests/cluster/provision_master.sh
    - source_hash: https://raw.githubusercontent.com/rombie/contrail-kubernetes/manifests/cluster/manifests.hash
    - cwd: /
    - user: root
    - group: root
    - mode: 755
    - shell: /bin/bash
