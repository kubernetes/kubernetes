etcdctl-install:
  git.latest:
    - target: /var/src/etcdctl
    - name: git://github.com/coreos/etcdctl
  cmd.wait:
    - cwd: /var/src/etcdctl
    - name: ./build
    - env:
      - PATH: {{ grains['path'] }}:/usr/local/bin
    - watch:
      - git: etcdctl-install
  file.symlink:
    - name: /usr/local/bin/etcdctl
    - target: /var/src/etcdctl/bin/etcdctl
    - watch:
      - cmd: etcdctl-install
