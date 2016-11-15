# Early configurations of Kubernetes ran etcd on the host and as part of a migration step, we began to delete the host etcd
# It's possible though that the host has configured a separate etcd to configure other services like Flannel
# In that case, we do not want Salt to remove or stop the host service
# Note: its imperative that the host installed etcd not conflict with the Kubernetes managed etcd
{% if grains['keep_host_etcd'] is not defined %}

delete_etc_etcd_dir:
  file.absent:
    - name: /etc/etcd

delete_etcd_conf:
  file.absent:
    - name: /etc/etcd/etcd.conf

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

#stop legacy etcd_service
stop_etcd-service:
  service.dead:
    - name: etcd
    - enable: None

{% endif %}

touch /var/log/etcd.log:
  cmd.run:
    - creates: /var/log/etcd.log

touch /var/log/etcd-events.log:
  cmd.run:
    - creates: /var/log/etcd-events.log

/var/etcd:
  file.directory:
    - user: root
    - group: root
    - dir_mode: 700
    - recurse:
      - user
      - group
      - mode

/etc/kubernetes/manifests/etcd.manifest:
  file.managed:
    - source: salt://etcd/etcd.manifest
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - makedirs: true
    - dir_mode: 755
    - context:
        suffix: ""
        port: 2379
        server_port: 2380
        cpulimit: '"200m"'

/etc/kubernetes/manifests/etcd-events.manifest:
  file.managed:
    - source: salt://etcd/etcd.manifest
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - makedirs: true
    - dir_mode: 755
    - context:
        suffix: "-events"
        port: 4002
        server_port: 2381
        cpulimit: '"100m"'
