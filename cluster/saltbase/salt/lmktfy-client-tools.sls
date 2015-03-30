/usr/local/bin/lmktfyctl:
  file.managed:
    - source: salt://lmktfy-bins/lmktfyctl
    - user: root
    - group: root
    - mode: 755
