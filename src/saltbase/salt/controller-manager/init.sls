{% set root = '/var/src/controller-manager' %}
{% set package = 'github.com/GoogleCloudPlatform/kubernetes' %}
{% set package_dir = root + '/src/' + package %}

{{ package_dir }}:
  file.recurse:
    - source: salt://controller-manager/go
    - user: root
    - group: staff
    - dir_mode: 775
    - file_mode: 664
    - makedirs: True
    - recurse:
      - user
      - group
      - mode

controller-manager-third-party-go:
  file.recurse:
    - name: {{ root }}/src
    - source: salt://third-party/go/src
    - user: root
    - group: staff
    - dir_mode: 775
    - file_mode: 664
    - makedirs: True
    - recurse:
      - user
      - group
      - mode

/etc/default/controller-manager:
  file.managed:
    - source: salt://controller-manager/default
    - template: jinja
    - user: root
    - group: root
    - mode: 644

controller-manager-build:
  cmd.wait:
    - cwd: {{ root }}
    - names:
      - go build {{ package }}/cmd/controller-manager
    - env:
      - PATH: {{ grains['path'] }}:/usr/local/bin
      - GOPATH: {{ root }}
    - watch:
      - file: {{ package_dir }}

/usr/local/bin/controller-manager:
  file.symlink:
    - target: {{ root }}/controller-manager
    - watch:
      - cmd: controller-manager-build

/etc/init.d/controller-manager:
  file.managed:
    - source: salt://controller-manager/initd
    - user: root
    - group: root
    - mode: 755

controller-manager:
  group.present:
    - system: True
  user.present:
    - system: True
    - gid_from_name: True
    - shell: /sbin/nologin
    - home: /var/controller-manager
    - require:
      - group: controller-manager
  service.running:
    - enable: True
    - watch:
      - cmd: controller-manager-build
      - file: /usr/local/bin/controller-manager
      - file: /etc/init.d/controller-manager
      - file: /etc/default/controller-manager

