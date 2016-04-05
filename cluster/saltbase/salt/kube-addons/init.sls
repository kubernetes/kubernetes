addon-dir-delete:
  file.absent:
    - name: /etc/kubernetes/addons

addon-dir-create:
  file.directory:
    - name: /etc/kubernetes/addons
    - user: root
    - group: root
    - mode: 0755
    - require:
        - file: addon-dir-delete

/etc/kubernetes/addons/namespace.yaml:
  file.managed:
    - source: salt://kube-addons/namespace.yaml
    - user: root
    - group: root
    - file_mode: 644

{% if pillar.get('enable_cluster_monitoring', '').lower() == 'influxdb' %}
/etc/kubernetes/addons/cluster-monitoring/influxdb:
  file.recurse:
    - source: salt://kube-addons/cluster-monitoring/influxdb
    - include_pat: E@(^.+\.yaml$|^.+\.json$)
    - template: jinja
    - user: root
    - group: root
    - dir_mode: 755
    - file_mode: 644
{% endif %}

{% if pillar.get('enable_l7_loadbalancing', '').lower() == 'glbc' %}
/etc/kubernetes/addons/cluster-loadbalancing/glbc:
  file.recurse:
    - source: salt://kube-addons/cluster-loadbalancing/glbc
    - include_pat: E@(^.+\.yaml$|^.+\.json$)
    - template: jinja
    - user: root
    - group: root
    - dir_mode: 755
    - file_mode: 644
{% endif %}

{% if pillar.get('enable_cluster_monitoring', '').lower() == 'google' %}
/etc/kubernetes/addons/cluster-monitoring/google:
  file.recurse:
    - source: salt://kube-addons/cluster-monitoring/google
    - include_pat: E@(^.+\.yaml$|^.+\.json$)
    - template: jinja
    - user: root
    - group: root
    - dir_mode: 755
    - file_mode: 644
{% endif %}

{% if pillar.get('enable_cluster_monitoring', '').lower() == 'standalone' %}
/etc/kubernetes/addons/cluster-monitoring/standalone:
  file.recurse:
    - source: salt://kube-addons/cluster-monitoring/standalone
    - include_pat: E@(^.+\.yaml$|^.+\.json$)
    - template: jinja
    - user: root
    - group: root
    - dir_mode: 755
    - file_mode: 644
{% endif %}

{% if pillar.get('enable_cluster_monitoring', '').lower() == 'googleinfluxdb' %}
/etc/kubernetes/addons/cluster-monitoring/googleinfluxdb:
  file.recurse:
    - source: salt://kube-addons/cluster-monitoring
    - include_pat: E@(^.+\.yaml$|^.+\.json$)
    - exclude_pat: E@(^.+heapster-controller\.yaml$|^.+heapster-controller\.json$)
    - template: jinja
    - user: root
    - group: root
    - dir_mode: 755
    - file_mode: 644
{% endif %}

{% if pillar.get('enable_cluster_dns', '').lower() == 'true' %}
/etc/kubernetes/addons/dns/skydns-svc.yaml:
  file.managed:
    - source: salt://kube-addons/dns/skydns-svc.yaml.in
    - template: jinja
    - group: root
    - dir_mode: 755
    - makedirs: True

/etc/kubernetes/addons/dns/skydns-rc.yaml:
  file.managed:
    - source: salt://kube-addons/dns/skydns-rc.yaml.in
    - template: jinja
    - group: root
    - dir_mode: 755
    - makedirs: True
{% endif %}

{% if pillar.get('enable_cluster_registry', '').lower() == 'true' %}
/etc/kubernetes/addons/registry/registry-svc.yaml:
  file.managed:
    - source: salt://kube-addons/registry/registry-svc.yaml
    - user: root
    - group: root
    - file_mode: 644
    - makedirs: True

/etc/kubernetes/addons/registry/registry-rc.yaml:
  file.managed:
    - source: salt://kube-addons/registry/registry-rc.yaml
    - user: root
    - group: root
    - file_mode: 644
    - makedirs: True

/etc/kubernetes/addons/registry/registry-pv.yaml:
  file.managed:
    - source: salt://kube-addons/registry/registry-pv.yaml.in
    - template: jinja
    - user: root
    - group: root
    - file_mode: 644
    - makedirs: True

/etc/kubernetes/addons/registry/registry-pvc.yaml:
  file.managed:
    - source: salt://kube-addons/registry/registry-pvc.yaml.in
    - template: jinja
    - user: root
    - group: root
    - file_mode: 644
    - makedirs: True
{% endif %}

{% if pillar.get('enable_node_logging', '').lower() == 'true'
   and pillar.get('logging_destination', '').lower() == 'elasticsearch'
   and pillar.get('enable_cluster_logging', '').lower() == 'true' %}
/etc/kubernetes/addons/fluentd-elasticsearch:
  file.recurse:
    - source: salt://kube-addons/fluentd-elasticsearch
    - include_pat: E@^.+\.yaml$
    - user: root
    - group: root
    - dir_mode: 755
    - file_mode: 644
{% endif %}

{% if pillar.get('enable_cluster_ui', '').lower() == 'true' %}
/etc/kubernetes/addons/dashboard:
  file.recurse:
    - source: salt://kube-addons/dashboard
    - include_pat: E@^.+\.yaml$
    - user: root
    - group: root
    - dir_mode: 755
    - file_mode: 644
{% endif %}

/etc/kubernetes/kube-addons.sh:
  file.managed:
    - source: salt://kube-addons/kube-addons.sh
    - user: root
    - group: root
    - mode: 755

/etc/kubernetes/kube-addon-update.sh:
  file.managed:
    - source: salt://kube-addons/kube-addon-update.sh
    - user: root
    - group: root
    - mode: 755

{% if pillar.get('is_systemd') %}

{{ pillar.get('systemd_system_path') }}/kube-addons.service:
  file.managed:
    - source: salt://kube-addons/kube-addons.service
    - user: root
    - group: root
  cmd.wait:
    - name: /opt/kubernetes/helpers/services bounce kube-addons
    - watch:
      - file: {{ pillar.get('systemd_system_path') }}/kube-addons.service

{% else %}

/etc/init.d/kube-addons:
  file.managed:
    - source: salt://kube-addons/initd
    - user: root
    - group: root
    - mode: 755

{% endif %}

# Stop kube-addons service each time salt is executed, just in case
# there was a modification of addons.
# Actually, this should be handled by watching file changes, but
# somehow it doesn't work.
service-kube-addon-stop:
  service.dead:
    - name: kube-addons

kube-addons:
  service.running:
    - enable: True
    - require:
        - service: service-kube-addon-stop
    - watch:
{% if pillar.get('is_systemd') %}
      - file: {{ pillar.get('systemd_system_path') }}/kube-addons.service
{% else %}
      - file: /etc/init.d/kube-addons
{% endif %}
{% if pillar.get('is_systemd') %}
    - provider:
      - service: systemd
{%- endif %}
