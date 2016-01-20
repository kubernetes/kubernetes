touch /var/log/flannel.log:
  cmd.run:
    - creates: /var/log/flannel.log

touch /var/log/etcd_flannel.log:
  cmd.run:
    - creates: /var/log/etcd_flannel.log

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
