{% if grains.kubelet_api_servers is not defined -%}
/etc/kubernetes/manifests/fluentd-gcp.yaml:
  file.managed:
    - source: salt://fluentd-gcp/fluentd-gcp.yaml
    - user: root
    - group: root
    - mode: 644
    - makedirs: true
    - dir_mode: 755
{% endif %}