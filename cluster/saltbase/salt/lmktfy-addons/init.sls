{% if pillar.get('enable_cluster_monitoring', '').lower() == 'true' %}
/etc/lmktfyrnetes/addons/cluster-monitoring:
  file.recurse:
    - source: salt://lmktfy-addons/cluster-monitoring
    - include_pat: E@^.+\.yaml$
    - user: root
    - group: root
    - dir_mode: 755
    - file_mode: 644
{% endif %}

{% if pillar.get('enable_cluster_dns', '').lower() == 'true' %}
/etc/lmktfyrnetes/addons/dns/skydns-svc.yaml:
  file.managed:
    - source: salt://lmktfy-addons/dns/skydns-svc.yaml.in
    - template: jinja
    - group: root
    - dir_mode: 755
    - makedirs: True

/etc/lmktfyrnetes/addons/dns/skydns-rc.yaml:
  file.managed:
    - source: salt://lmktfy-addons/dns/skydns-rc.yaml.in
    - template: jinja
    - group: root
    - dir_mode: 755
    - makedirs: True
{% endif %}

{% if pillar.get('enable_node_logging', '').lower() == 'true'
   and pillar.get('logging_destination').lower() == 'elasticsearch'
   and pillar.get('enable_cluster_logging', '').lower() == 'true' %}
/etc/lmktfyrnetes/addons/fluentd-elasticsearch:
  file.recurse:
    - source: salt://lmktfy-addons/fluentd-elasticsearch
    - include_pat: E@^.+\.yaml$
    - user: root
    - group: root
    - dir_mode: 755
    - file_mode: 644

/etc/lmktfyrnetes/addons/fluentd-elasticsearch/es-controller.yaml:
  file.managed:
    - source: salt://lmktfy-addons/fluentd-elasticsearch/es-controller.yaml.in
    - template: jinja
    - group: root
    - dir_mode: 755
    - makedirs: True
{% endif %}

{% if grains['os_family'] == 'RedHat' %}

/usr/lib/systemd/system/lmktfy-addons.service:
  file.managed:
    - source: salt://lmktfy-addons/lmktfy-addons.service
    - user: root
    - group: root

/etc/lmktfyrnetes/lmktfy-addons.sh:
  file.managed:
    - source: salt://lmktfy-addons/lmktfy-addons.sh
    - user: root
    - group: root
    - mode: 755

{% else %}

/etc/init.d/lmktfy-addons:
  file.managed:
    - source: salt://lmktfy-addons/initd
    - user: root
    - group: root
    - mode: 755

{% endif %}

lmktfy-addons:
  service.running:
    - enable: True
