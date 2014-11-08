/usr/local/bin/kubecfg:
  file.managed:
    - source: salt://kube-bins/kubecfg
    - user: root
    - group: root
    - mode: 755

/usr/local/bin/kubectl:
  file.managed:
    - source: salt://kube-bins/kubectl
    - user: root
    - group: root
    - mode: 755
