{% if grains.cloud is defined %}
  {% if grains.cloud == 'gce' %}
    {% set cert_ip='_use_gce_external_ip_' %}
  {% endif %}
  {% if grains.cloud == 'aws' %}
    {% set cert_ip='_use_aws_external_ip_' %}
  {% endif %}
  {% if grains.cloud == 'azure' %}
    {% set cert_ip='_use_azure_dns_name_' %}
  {% endif %}
  {% if grains.cloud == 'vagrant' %}
    {% set cert_ip=grains.ip_interfaces.eth1[0] %}
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

lmktfy-cert:
  group.present:
    - system: True

lmktfyrnetes-cert:
  cmd.script:
    - unless: test -f /srv/lmktfyrnetes/server.cert
    - source: salt://generate-cert/{{certgen}}
{% if cert_ip is defined %}
    - args: {{cert_ip}}
    - require:
      - pkg: curl
{% endif %}
    - cwd: /
    - user: root
    - group: root
    - shell: /bin/bash

# These are introduced to ensure backwards compatability with older gcloud tools in GKE
lmktfyrnetes-old-key:
  file.copy:
    - name: /usr/share/nginx/lmktfycfg.key
    - source: /srv/lmktfyrnetes/lmktfycfg.key
    - makedirs: True
    - preserve: True
    - require:
      - cmd: lmktfyrnetes-cert


lmktfyrnetes-old-cert:
  file.copy:
    - name: /usr/share/nginx/lmktfycfg.crt
    - source: /srv/lmktfyrnetes/lmktfycfg.crt
    - makedirs: True
    - preserve: True
    - require:
      - cmd: lmktfyrnetes-cert


lmktfyrnetes-old-ca:
  file.copy:
    - name: /usr/share/nginx/ca.crt
    - source: /srv/lmktfyrnetes/ca.crt
    - makedirs: True
    - preserve: True
    - require:
      - cmd: lmktfyrnetes-cert
