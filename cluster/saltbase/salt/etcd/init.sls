{% set etcd_version="v0.4.6" %}
{% set etcd_tar_url="https://github.com/coreos/etcd/releases/download/%s/etcd-%s-linux-amd64.tar.gz"
  | format(etcd_version, etcd_version)  %}
{% set etcd_tar_hash="md5=661d58424ff33dd837b8ee988dd79ae3" %}

etcd-tar:
  archive:
    - extracted
    - user: root
    - name: /usr/local/src
    - source: {{ etcd_tar_url }}
    - source_hash: {{ etcd_tar_hash }}
    - archive_format: tar
    - if_missing: /usr/local/src/etcd-{{ etcd_version }}-linux-amd64
    - tar_options: z
  file.directory:
    - name: /usr/local/src/etcd-{{ etcd_version }}-linux-amd64
    - user: root
    - group: root
    - watch:
      - archive: etcd-tar
    - recurse:
      - user
      - group

etcd-symlink:
  file.symlink:
    - name: /usr/local/bin/etcd
    - target: /usr/local/src/etcd-{{ etcd_version }}-linux-amd64/etcd
    - force: true
    - watch:
      - archive: etcd-tar

etcdctl-symlink:
  file.symlink:
    - name: /usr/local/bin/etcdctl
    - target: /usr/local/src/etcd-{{ etcd_version }}-linux-amd64/etcdctl
    - force: true
    - watch:
      - archive: etcd-tar

etcd:
  group.present:
    - system: True
  user.present:
    - system: True
    - gid_from_name: True
    - shell: /sbin/nologin
    - home: /var/etcd

/etc/etcd:
  file.directory:
    - user: root
    - group: root
    - dir_mode: 755

/etc/etcd/etcd.conf:
  file.managed:
    - source: salt://etcd/etcd.conf
    - user: root
    - group: root
    - mode: 644

/var/etcd:
  file.directory:
    - user: etcd
    - group: etcd
    - dir_mode: 700

{% if grains['os_family'] == 'RedHat' %}

/etc/default/etcd:
  file.managed:
    - source: salt://etcd/default
    - template: jinja
    - user: root
    - group: root
    - mode: 644

/usr/lib/systemd/system/etcd.service:
  file.managed:
    - source: salt://etcd/etcd.service
    - user: root
    - group: root

{% else %}

/etc/init.d/etcd:
  file.managed:
    - source: salt://etcd/initd
    - user: root
    - group: root
    - mode: 755

{% endif %}

etcd-service:
  service.running:
    - name: etcd
    - enable: True
    - watch:
      - file: /etc/etcd/etcd.conf
      {% if grains['os_family'] == 'RedHat' %}
      - file: /usr/lib/systemd/system/etcd.service
      - file: /etc/default/etcd
      {% endif %}
      - file: etcd-tar
      - file: etcd-symlink

