nginx:
  pkg:
    - installed
  service:
    - running
    - watch:
      - pkg: nginx
      - file: /etc/nginx/nginx.conf
      - file: /etc/nginx/sites-enabled/default
      - file: /usr/share/nginx/htpasswd
      - cmd: /usr/share/nginx/server.cert

{% if grains.cloud is defined %}
  {% if grains.cloud == 'gce' %}
    {% set cert_ip='_use_gce_external_ip_' %}
  {% endif %}
  {% if grains.cloud == 'aws' %}
    {% set cert_ip='_use_aws_external_ip_' %}
  {% endif %}
  {% if grains.cloud == 'vagrant' %}
    {% set cert_ip=grains.fqdn_ip4 %}
  {% endif %}
  {% if grains.cloud == 'vsphere' %}
    {% set cert_ip=grains.ip_interfaces.eth0[0] %}
  {% endif %}
{% endif %}
# If there is a pillar defined, override any defaults.
{% if pillar['cert_ip'] is defined %}
  {% set cert_ip=pillar['cert_ip'] %}
{% endif %}

{% set certgen="make-cert.sh" %}
{% if cert_ip is defined %}
  {% set certgen="make-ca-cert.sh" %}
{% endif %}

/usr/share/nginx/server.cert:
  cmd.script:
    - unless: test -f /usr/share/nginx/server.cert
    - source: salt://nginx/{{certgen}} 
{% if cert_ip is defined %}
    - args: {{cert_ip}}
    - require:
      - pkg: curl
{% endif %}
    - cwd: /
    - user: root
    - group: root
    - shell: /bin/bash

/etc/nginx/nginx.conf:
  file:
    - managed
    - source: salt://nginx/nginx.conf
    - template: jinja
    - user: root
    - group: root
    - mode: 644

/etc/nginx/sites-enabled/default:
  file:
    - managed
    - makedirs: true
    - source: salt://nginx/kubernetes-site
    - user: root
    - group: root
    - mode: 644

/usr/share/nginx/htpasswd:
  file:
    - managed
    - source: salt://nginx/htpasswd
    - user: root
    - group: root
    - mode: 644

