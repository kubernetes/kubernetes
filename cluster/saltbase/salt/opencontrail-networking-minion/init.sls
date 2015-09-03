opencontrail-networking-minion:
  cmd.script:
    - unless: test -f /var/log/contrail/provision_minion.log
    - source: https://raw.githubusercontent.com/juniper/contrail-kubernetes/master/cluster/provision_minion.sh
    - source_hash: https://raw.githubusercontent.com/juniper/contrail-kubernetes/master/cluster/manifests.hash
    - cwd: /
    - user: root
    - group: root
    - mode: 755
    - shell: /bin/bash
