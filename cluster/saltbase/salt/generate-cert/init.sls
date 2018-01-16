{% set master_extra_sans=grains.get('master_extra_sans', '') %}
{% if grains.cloud is defined and grains.cloud == 'gce' %}
  {% set cert_ip='_use_gce_external_ip_' %}
{% endif %}

# If there is a pillar defined, override any defaults.
{% if pillar['cert_ip'] is defined %}
  {% set cert_ip=pillar['cert_ip'] %}
{% endif %}

{% set certgen="make-cert.sh" %}
{% if cert_ip is defined %}
  {% set certgen="make-ca-cert.sh" %}
{% endif %}

openssl:
  pkg.installed: []

kube-cert:
  group.present:
    - system: True

kubernetes-cert:
  cmd.script:
    - unless: test -f /srv/kubernetes/server.cert
    - source: salt://generate-cert/{{certgen}}
{% if cert_ip is defined %}
    - args: {{cert_ip}} {{master_extra_sans}}
    - require:
      - pkg: curl
{% endif %}
    - cwd: /
    - user: root
    - group: root
    - shell: /bin/bash
    - require:
      - pkg: openssl
