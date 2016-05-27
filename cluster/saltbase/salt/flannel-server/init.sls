touch /var/log/flannel.log:
  cmd.run:
    - creates: /var/log/flannel.log

touch /var/log/etcd_flannel.log:
  cmd.run:
    - creates: /var/log/etcd_flannel.log

/var/etcd-flannel:
  file.directory:
    - user: root
    - group: root
    - dir_mode: 700
    - recurse:
      - user
      - group
      - mode

/etc/kubernetes/network.json:
  file.managed:
    - source: salt://flannel-server/network.json
    - makedirs: True
    - user: root
    - group: root
    - mode: 755

/etc/kubernetes/manifests/flannel-server.manifest:
  file.managed:
    - source: salt://flannel-server/flannel-server.manifest
    - user: root
    - group: root
    - mode: 644
    - makedirs: true
    - dir_mode: 755
    - template: jinja
    - context:
        etcd_port: 4003
        etcd_peer_port: 2382
        cpulimit: '"100m"'
