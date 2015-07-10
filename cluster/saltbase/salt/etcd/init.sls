delete_etc_etcd_dir:
  file.absent:
    - name: /etc/etcd

delete_etcd_conf:
  file.absent:
    - name: /etc/etcd/etcd.conf

touch /var/log/etcd.log:
  cmd.run:
    - creates: /var/log/etcd.log

/var/etcd:
  file.directory:
    - user: root
    - group: root
    - dir_mode: 700
    - recurse:
      - user
      - group
      - mode

delete_etcd_default:
  file.absent:
    - name: /etc/default/etcd

{% if pillar.get('is_systemd') %}
delete_etcd_service_file:
  file.absent:
    - name: {{ pillar.get('systemd_system_path') }}/etcd.service
{% endif %}

delete_etcd_initd:
  file.absent:
    - name: /etc/init.d/etcd

/etc/kubernetes/manifests/etcd.manifest:
  file.managed:
    - source: salt://etcd/etcd.manifest
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - makedirs: true
    - dir_mode: 755

#stop legacy etcd_service 
stop_etcd-service:
  service.dead:
    - name: etcd
    - enable: None
