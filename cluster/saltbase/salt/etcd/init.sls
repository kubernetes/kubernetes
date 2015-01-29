# We are caching the etcd tar file in GCS for reliability and speed.  To
# update this to a new version, do the following:
# 2. Download tar file:
#    curl -LO https://github.com/coreos/etcd/releases/download/<ver>/etcd-<ver>-linux-amd64.tar.gz
# 3. Upload to GCS (the cache control makes :
#    gsutil cp <tar> gs://kubernetes-release/etcd/<tar>
# 4. Make it world readable:
#    gsutil -m acl ch -R -g all:R gs://kubernetes-release/etcd/
# 5. Get a hash of the tar:
#    shasum <tar>
# 6. Update this file with new tar version and new hash

{% set etcd_version="v0.4.6" %}
{% set etcd_tar_url="https://storage.googleapis.com/kubernetes-release/etcd/etcd-%s-linux-amd64.tar.gz"
  | format(etcd_version)  %}
{% set etcd_tar_hash="sha1=5db514e30b9f340eda00671230d5136855ae14d7" %}

etcd-tar:
  archive:
    - extracted
    - user: root
    - name: /usr/local/src
    - source: {{ etcd_tar_url }}
    - source_hash: {{ etcd_tar_hash }}
    - archive_format: tar
    - if_missing: /usr/local/src/etcd-{{ etcd_version }}-linux-amd64
{% if grains['saltversioninfo'] <= (2014, 7, 0, 0) %}
    - tar_options: xz
{% endif %}
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
    - require:
      - user: etcd
      - group: etcd

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
    - require:
      - file: /var/etcd
      - user: etcd
      - group: etcd

