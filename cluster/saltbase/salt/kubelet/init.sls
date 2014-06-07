{% set root = '/var/src/kubelet' %}
{% set package = 'github.com/GoogleCloudPlatform/kubernetes' %}
{% set package_dir = root + '/src/' + package %}

{{ package_dir }}:
  file.recurse:
    - source: salt://kubelet/go
    - user: root
    - group: staff
    - dir_mode: 775
    - file_mode: 664
    - makedirs: True
    - recurse:
      - user
      - group
      - mode

kubelet-third-party-go:
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

/etc/default/kubelet:
  file.managed:
    - source: salt://kubelet/default
    - template: jinja
    - user: root
    - group: root
    - mode: 644

kubelet-build:
  cmd.wait:
    - cwd: {{ root }}
    - names:
      - go build {{ package }}/cmd/kubelet
    - env:
      - PATH: {{ grains['path'] }}:/usr/local/bin
      - GOPATH: {{ root }}
    - watch:
      - file: {{ package_dir }}

/usr/local/bin/kubelet:
  file.symlink:
    - target: {{ root }}/kubelet
    - watch:
      - cmd: kubelet-build

/etc/init.d/kubelet:
  file.managed:
    - source: salt://kubelet/initd
    - user: root
    - group: root
    - mode: 755

kubelet:
  group.present:
    - system: True
  user.present:
    - system: True
    - gid_from_name: True
    - shell: /sbin/nologin
    - home: /var/kubelet
    - groups:
      - docker
    - require:
      - group: kubelet
  service.running:
    - enable: True
    - watch:
      - cmd: kubelet-build
      - file: /usr/local/bin/kubelet
      - file: /etc/init.d/kubelet

