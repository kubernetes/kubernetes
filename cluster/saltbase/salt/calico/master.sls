{% if grains.network_mode is defined and grains.network_mode == 'calico' %}

/etc/kubernetes/manifests/calico-etcd.manifest:
  file.managed:
    - source: salt://calico/calico-etcd.manifest
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - makedirs: true
    - dir_mode: 755

{% endif %}
