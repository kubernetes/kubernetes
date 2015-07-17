{% if grains.network_mode is defined and grains.network_mode == 'calico' %}

include:
  - docker

calicoctl:
  file.managed:
    - name: /home/vagrant/calicoctl
    - source: https://github.com/Metaswitch/calico-docker/releases/download/v0.4.8/calicoctl
    - source_hash: sha512=814fd7369ba395c67e35245115a5885d1722300301d32585f9003f63e94cd26f77e325ae765ba2f6cba2fddec5ffdb8e4f7bc1b326f9dc343cf03e96b77a679e
    - makedirs: True
    - mode: 744

calico-network-plugin:
  file.managed:
    - name: /usr/libexec/kubernetes/kubelet-plugins/net/exec/calico/calico
    - source: https://github.com/Metaswitch/calico-docker/releases/download/v0.4.8/calico_kubernetes
    - source_hash: sha512=50a8b9ca38d6813a4793c76959acc80d2d38002547a0f5aa0f974fc88407dd7c09abfb6ec1c30ae76690740cf36910cd89d1a1b9786679929f65bd0538fa292f
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

calico-ip-pool-reset:
  cmd.run:
    - name: /home/vagrant/calicoctl pool remove 192.168.0.0/16
    - env:
      - ETCD_AUTHORITY: "{{ grains.api_servers }}:6666"
    - require:
      - service: docker
      - file: calicoctl
      - cmd: calico-node

calico-ip-pool:
  cmd.run:
    - name: /home/vagrant/calicoctl pool add {{ grains['cbr-cidr'] }}
    - env:
      - ETCD_AUTHORITY: "{{ grains.api_servers }}:6666"
    - require:
      - service: docker
      - file: calicoctl
      - cmd: calico-node

ip6_tables:
  kmod.present

xt_set:
  kmod.present

{% endif %}
