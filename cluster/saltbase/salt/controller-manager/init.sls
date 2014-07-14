{% set root = '/var/src/controller-manager' %}
{% set package = 'github.com/GoogleCloudPlatform/kubernetes' %}
{% set package_dir = root + '/src/' + package %}
{% if grains['os_family'] == 'RedHat' %}
{% set environment_file = '/etc/sysconfig/controller-manager' %}
{% else %}
{% set environment_file = '/etc/default/controller-manager' %}
{% endif %}

{{ package_dir }}:
  file.recurse:
    - source: salt://controller-manager/go
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

controller-manager-third-party-go:
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

{{ environment_file }}:
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

{% if grains['os_family'] == 'RedHat' %}

/usr/lib/systemd/system/controller-manager.service:
  file.managed:
    - source: salt://controller-manager/controller-manager.service
    - user: root
    - group: root

{% else %}

/etc/init.d/controller-manager:
  file.managed:
    - source: salt://controller-manager/initd
    - user: root
    - group: root
    - mode: 755

{% endif %}

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
      - file: {{ environment_file }}
{% if grains['os_family'] != 'RedHat' %}
      - file: /etc/init.d/controller-manager
{% endif %}


