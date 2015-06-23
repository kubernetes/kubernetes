/usr/local/bin/kubectl:
  file.managed:
    - source: salt://kube-bins/kubectl
    - user: root
    - group: root
    - mode: 755
/usr/local/bin/resource-query:
  file.managed:
    - source: salt://kube-bins/resource-query
    - user: root
    - group: root
    - mode: 755
