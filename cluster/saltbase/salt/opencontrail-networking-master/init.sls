opencontrail-networking-master:
  cmd.script:
    - source: salt://opencontrail-networking-master/provision.sh
    - cwd: /
    - user: root
    - group: root
    - mode: 755
    - shell: /bin/bash
