{% set root = '/var/src/kube-proxy' %}
{% set package = 'github.com/GoogleCloudPlatform/kubernetes' %}
{% set package_dir = root + '/src/' + package %}

{{ package_dir }}:
  file.recurse:
    - source: salt://kube-proxy/go
    - user: root
    - group: staff
    - dir_mode: 775
    - file_mode: 664
    - makedirs: True
    - recurse:
      - user
      - group
      - mode

third-party-go:
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

kube-proxy-build:
  cmd.wait:
    - cwd: {{ root }}
    - names:
      - go build {{ package }}/cmd/proxy
    - env:
      - PATH: {{ grains['path'] }}:/usr/local/bin
      - GOPATH: {{ root }}
    - watch:
      - file: {{ package_dir }}

/usr/local/bin/kube-proxy:
  file.symlink:
    - target: {{ root }}/proxy
    - watch:
      - cmd: kube-proxy-build

/etc/init.d/kube-proxy:
  file.managed:
    - source: salt://kube-proxy/initd
    - user: root
    - group: root
    - mode: 755

/etc/default/kube-proxy:
  file.managed:
    - source: salt://kube-proxy/default
    - template: jinja
    - user: root
    - group: root
    - mode: 644

kube-proxy:
  group.present:
    - system: True
  user.present:
    - system: True
    - gid_from_name: True
    - shell: /sbin/nologin
    - home: /var/kube-proxy
    - require:
      - group: kube-proxy
  service.running:
    - enable: True
    - watch:
      - cmd: kube-proxy-build
      - file: /etc/default/kube-proxy
      - file: /etc/init.d/kube-proxy
