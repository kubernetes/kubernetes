{% set root = '/var/src/kube-proxy' %}
{% set package = 'github.com/GoogleCloudPlatform/kubernetes' %}
{% set package_dir = root + '/src/' + package %}
{% if grains['os_family'] == 'RedHat' %}
{% set environment_file = '/etc/sysconfig/kube-proxy' %}
{% else %}
{% set environment_file = '/etc/default/kube-proxy' %}
{% endif %}

{{ package_dir }}:
  file.recurse:
    - source: salt://kube-proxy/go
    - user: root
    {% if grains['os_family'] == 'RedHat' %}
    - group: root
    {% else %}
    - group: staff
    {% endif %}
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
    {% if grains['os_family'] == 'RedHat' %}
    - group: root
    {% else %}
    - group: staff
    {% endif %}
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

{% if grains['os_family'] == 'RedHat' %}

/usr/lib/systemd/system/kube-proxy.service:
  file.managed:
    - source: salt://kube-proxy/kube-proxy.service
    - user: root
    - group: root

{% else %}

/etc/init.d/kube-proxy:
  file.managed:
    - source: salt://kube-proxy/initd
    - user: root
    - group: root
    - mode: 755

{% endif %}

{{ environment_file }}:
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
      - file: {{ environment_file }}
{% if grains['os_family'] != 'RedHat' %}
      - file: /etc/init.d/kube-proxy
{% endif %}
