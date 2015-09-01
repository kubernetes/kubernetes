opencontrail-networking-minion:
  cmd.script:
    - source: salt://opencontrail-networking-minion/provision.sh
    - cwd: /
    - user: root
    - group: root
    - mode: 755
    - shell: /bin/bash
