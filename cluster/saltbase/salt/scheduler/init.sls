{% set root = '/var/src/scheduler' %}
{% set package = 'github.com/GoogleCloudPlatform/kubernetes' %}
{% set package_dir = root + '/src/' + package %}
{% set go_opt = pillar['go_opt'] %}
{% if grains['os_family'] == 'RedHat' %}
{% set environment_file = '/etc/sysconfig/scheduler' %}
{% else %}
{% set environment_file = '/etc/default/scheduler' %}
{% endif %}

{{ package_dir }}:
  file.recurse:
    - source: salt://scheduler/go
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
    - source: salt://scheduler/default
    - template: jinja
    - user: root
    - group: root
    - mode: 644

scheduler-build:
  cmd.run:
    - cwd: {{ root }}
    - names:
      - go build {{ go_opt }} {{ package }}/plugin/cmd/scheduler
    - env:
      - PATH: {{ grains['path'] }}:/usr/local/bin
      - GOPATH: {{ root }}:{{ package_dir }}/Godeps/_workspace
    - require:
      - file: {{ package_dir }}

/usr/local/bin/scheduler:
  file.symlink:
    - target: {{ root }}/scheduler
    - watch:
      - cmd: scheduler-build

{% if grains['os_family'] == 'RedHat' %}

/usr/lib/systemd/system/scheduler.service:
  file.managed:
    - source: salt://scheduler/scheduler.service
    - user: root
    - group: root

{% else %}

/etc/init.d/scheduler:
  file.managed:
    - source: salt://scheduler/initd
    - user: root
    - group: root
    - mode: 755

{% endif %}

scheduler:
  group.present:
    - system: True
  user.present:
    - system: True
    - gid_from_name: True
    - shell: /sbin/nologin
    - home: /var/scheduler
    - require:
      - group: scheduler
  service.running:
    - enable: True
    - watch:
      - cmd: scheduler-build
      - file: /usr/local/bin/scheduler
      - file: {{ environment_file }}
{% if grains['os_family'] != 'RedHat' %}
      - file: /etc/init.d/scheduler
{% endif %}


