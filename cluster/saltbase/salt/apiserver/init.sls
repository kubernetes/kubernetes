{% set root = '/var/src/apiserver' %}
{% set package = 'github.com/GoogleCloudPlatform/kubernetes' %}
{% set package_dir = root + '/src/' + package %}
{% set go_opt = pillar['go_opt'] %}
{% if grains['os_family'] == 'RedHat' %}
{% set environment_file = '/etc/sysconfig/apiserver' %}
{% else %}
{% set environment_file = '/etc/default/apiserver' %}
{% endif %}

{{ package_dir }}:
  file.recurse:
    - source: salt://apiserver/go
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
    - source: salt://apiserver/default
    - template: jinja
    - user: root
    - group: root
    - mode: 644

apiserver-build:
  cmd.run:
    - cwd: {{ root }}
    - names:
      - go build {{ go_opt }} {{ package }}/cmd/apiserver
    - env:
      - PATH: {{ grains['path'] }}:/usr/local/bin
      - GOPATH: {{ root }}:{{ package_dir }}/Godeps/_workspace
    - require:
      - file: {{ package_dir }}

/usr/local/bin/apiserver:
  file.symlink:
    - target: {{ root }}/apiserver
    - watch:
      - cmd: apiserver-build

{% if grains['os_family'] == 'RedHat' %}

/usr/lib/systemd/system/apiserver.service:
  file.managed:
    - source: salt://apiserver/apiserver.service
    - user: root
    - group: root

{% else %}

/etc/init.d/apiserver:
  file.managed:
    - source: salt://apiserver/initd
    - user: root
    - group: root
    - mode: 755

{% endif %}

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
      - file: {{ environment_file }}
      - file: /usr/local/bin/apiserver
{% if grains['os_family'] != 'RedHat' %}
      - file: /etc/init.d/apiserver
{% endif %}
