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

/etc/kubernetes/manifests/kube-addon-manager.yaml:
  file.managed:
    - source: salt://kube-addons/kube-addon-manager.yaml
    - user: root
    - group: root
    - mode: 755
