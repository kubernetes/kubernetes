{% if grains['os_family'] == 'RedHat' %}
{% set environment_file = '/etc/sysconfig/lmktfylet' %}
{% else %}
{% set environment_file = '/etc/default/lmktfylet' %}
{% endif %}

{{ environment_file}}:
  file.managed:
    - source: salt://lmktfylet/default
    - template: jinja
    - user: root
    - group: root
    - mode: 644

/usr/local/bin/lmktfylet:
  file.managed:
    - source: salt://lmktfy-bins/lmktfylet
    - user: root
    - group: root
    - mode: 755

{% if grains['os_family'] == 'RedHat' %}

/usr/lib/systemd/system/lmktfylet.service:
  file.managed:
    - source: salt://lmktfylet/lmktfylet.service
    - user: root
    - group: root

{% else %}

/etc/init.d/lmktfylet:
  file.managed:
    - source: salt://lmktfylet/initd
    - user: root
    - group: root
    - mode: 755

{% endif %}

# The default here is that this file is blank.  If this is the case, the lmktfylet
# won't be able to parse it as JSON and it'll not be able to publish events to
# the apiserver.  You'll see a single error line in the lmktfylet start up file
# about this.
/var/lib/lmktfylet/lmktfyrnetes_auth:
  file.managed:
    - source: salt://lmktfylet/lmktfyrnetes_auth
    - user: root
    - group: root
    - mode: 400
    - makedirs: true

lmktfylet:
  group.present:
    - system: True
  user.present:
    - system: True
    - gid_from_name: True
    - shell: /sbin/nologin
    - home: /var/lib/lmktfylet
{% if grains['os_family'] != 'RedHat' %}    
    - groups:
      - docker
{% endif %}      
    - require:
      - group: lmktfylet
  service.running:
    - enable: True
    - watch:
      - file: /usr/local/bin/lmktfylet
{% if grains['os_family'] != 'RedHat' %}
      - file: /etc/init.d/lmktfylet
{% endif %}
      - file: /var/lib/lmktfylet/lmktfyrnetes_auth
