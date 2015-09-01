opencontrail-networking-minion:
  cmd.script:
    - source: salt://opencontrail-networking-minion/provision.sh
    - cwd: /
    - user: root
    - group: root
    - shell: /bin/bash
