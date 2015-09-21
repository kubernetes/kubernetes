/usr/local/bin/kubectl:
  file.managed:
    - source: salt://kube-bins/kubectl
    - user: root
    - group: root
    - mode: 755
