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

calicoctl:
  file.managed:
    - name: /home/vagrant/calicoctl
    - source: https://github.com/Metaswitch/calico-docker/releases/download/v0.4.8/calicoctl
    - source_hash: sha512=814fd7369ba395c67e35245115a5885d1722300301d32585f9003f63e94cd26f77e325ae765ba2f6cba2fddec5ffdb8e4f7bc1b326f9dc343cf03e96b77a679e
    - makedirs: True
    - mode: 744

calico-node:
  cmd.run:
    - name: /home/vagrant/calicoctl node --ip={{ grains.node_ip }} --node-image=calico/node:v0.4.8
    - env:
      - ETCD_AUTHORITY: "{{ grains.api_servers }}:6666"
    - require:
      - kmod: ip6_tables
      - kmod: xt_set
      - service: docker
      - file: calicoctl

ip6_tables:
  kmod.present

xt_set:
  kmod.present

{% endif %}
