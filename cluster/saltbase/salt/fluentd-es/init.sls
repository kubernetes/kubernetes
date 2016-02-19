{% if grains['roles'][0] != 'kubernetes-master' -%}
/etc/kubernetes/manifests/fluentd-es.yaml:
  file.managed:
    - source: salt://fluentd-es/fluentd-es.yaml
    - user: root
    - group: root
    - mode: 644
    - makedirs: true
    - dir_mode: 755
{% endif %}
