etcd-install:
  git.latest:
    - target: /var/src/etcd
    - name: git://github.com/coreos/etcd
  cmd.wait:
    - cwd: /var/src/etcd
    - names:
      - ./build
    - env:
      - PATH: {{ grains['path'] }}:/usr/local/bin
    - watch:
      - git: etcd-install
  file.symlink:
    - name: /usr/local/bin/etcd
    - target: /var/src/etcd/bin/etcd
    - watch:
      - cmd: etcd-install

etcd:
  group.present:
    - system: True
  user.present:
    - system: True
    - gid_from_name: True
    - shell: /sbin/nologin
    - home: /var/etcd
    - require:
      - group: etcd

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
      - cmd: etcd-install

