{% set root = '/var/src/apiserver' %}
{% set package = 'github.com/GoogleCloudPlatform/kubernetes' %}
{% set package_dir = root + '/src/' + package %}

{{ package_dir }}:
  file.recurse:
    - source: salt://apiserver/go
    - user: root
    - group: staff
    - dir_mode: 775
    - file_mode: 664
    - makedirs: True
    - recurse:
      - user
      - group
      - mode

apiserver-third-party-go:
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

/etc/default/apiserver:
  file.managed:
    - source: salt://apiserver/default
    - template: jinja
    - user: root
    - group: root
    - mode: 644

apiserver-build:
  cmd.wait:
    - cwd: {{ root }}
    - names:
      - go build {{ package }}/cmd/apiserver
    - env:
      - PATH: {{ grains['path'] }}:/usr/local/bin
      - GOPATH: {{ root }}
    - watch:
      - file: {{ package_dir }}

/usr/local/bin/apiserver:
  file.symlink:
    - target: {{ root }}/apiserver
    - watch:
      - cmd: apiserver-build

/etc/init.d/apiserver:
  file.managed:
    - source: salt://apiserver/initd
    - user: root
    - group: root
    - mode: 755

apiserver:
  group.present:
    - system: True
  user.present:
    - system: True
    - gid_from_name: True
    - shell: /sbin/nologin
    - home: /var/apiserver
    - require:
      - group: apiserver
  service.running:
    - enable: True
    - watch:
      - cmd: apiserver-build
      - file: /etc/default/apiserver
      - file: /usr/local/bin/apiserver
      - file: /etc/init.d/apiserver

