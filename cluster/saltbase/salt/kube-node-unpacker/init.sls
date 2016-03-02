/etc/kubernetes/kube-node-unpacker.sh:
  file.managed:
    - source: salt://kube-node-unpacker/kube-node-unpacker.sh
    - makedirs: True
    - user: root
    - group: root
    - mode: 755

{% if grains.cloud is defined and grains.cloud == 'gce' %}
node-docker-image-tags:
  file.touch:
    - name: /srv/pillar/docker-images.sls
{% else %}
kube-proxy-tar:
  file.managed:
    - name: /srv/salt/kube-bins/kube-proxy.tar
    - source: salt://kube-bins/kube-proxy.tar
    - makedirs: True
    - user: root
    - group: root
    - mode: 644
{% endif %}

{% set is_helium = '0' %}
# Super annoying, the salt version on GCE is old enough that 'salt.cmd.run'
# isn't supported
{% if grains.cloud is defined and grains.cloud == 'aws' %}
   # Salt has terrible problems with systemd on AWS too
   {% set is_helium = '0' %}
{% endif %}
# Salt Helium doesn't support systemd modules for service running
{% if pillar.get('is_systemd') and is_helium == '0' %}

{{ pillar.get('systemd_system_path') }}/kube-node-unpacker.service:
  file.managed:
    - source: salt://kube-node-unpacker/kube-node-unpacker.service
    - user: root
    - group: root
  cmd.wait:
    - name: /opt/kubernetes/helpers/services bounce kube-node-unpacker
    - watch:
{% if grains.cloud is defined and grains.cloud == 'gce' %}
      - file: node-docker-image-tags
{% else %}
      - file: kube-proxy-tar
{% endif %}
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
{% if grains.cloud is defined and grains.cloud == 'gce' %}
      - file: node-docker-image-tags
{% else %}
      - file: kube-proxy-tar
{% endif %}
      - file: /etc/kubernetes/kube-node-unpacker.sh

{% endif %}
