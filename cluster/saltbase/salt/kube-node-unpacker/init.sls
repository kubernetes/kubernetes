/etc/kubernetes/kube-node-unpacker.sh:
  file.managed:
    - source: salt://kube-node-unpacker/kube-node-unpacker.sh
    - user: root
    - group: root
    - mode: 755

node-docker-image-tags:
  file.touch:
    - name: /srv/pillar/docker-images.sls

{% if pillar.get('is_systemd') %}

{{ pillar.get('systemd_system_path') }}/kube-node-unpacker.service:
  file.managed:
    - source: salt://kube-node-unpacker/kube-node-unpacker.service
    - user: root
    - group: root
  cmd.wait:
    - name: /opt/kubernetes/helpers/services bounce kube-node-unpacker
    - watch:
      - file: node-docker-image-tags
      - file: /etc/kubernetes/kube-node-unpacker.sh
      - file: {{ pillar.get('systemd_system_path') }}/kube-node-unpacker.service

{% else %}

/etc/init.d/kube-node-unpacker:
  file.managed:
    - source: salt://kube-node-unpacker/initd
    - user: root
    - group: root
    - mode: 755

kube-node-unpacker:
  service.running:
    - enable: True
    - restart: True
    - watch:
      - file: node-docker-image-tags
      - file: /etc/kubernetes/kube-node-unpacker.sh

{% endif %}
