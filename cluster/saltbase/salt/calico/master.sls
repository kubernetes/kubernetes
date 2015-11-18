{% if pillar.get('network_provider', '').lower() == 'calico' %}

calicoctl:
  file.managed:
    - name: /usr/bin/calicoctl
    - source: https://github.com/projectcalico/calico-docker/releases/download/v0.10.0/calicoctl
    - source_hash: sha512=5dd8110cebfc00622d49adddcccda9d4906e6bca8a777297e6c0ffbcf0f7e40b42b0d6955f2e04b457b0919cb2d5ce39d2a3255d34e6ba36e8350f50319b3896
    - makedirs: True
    - mode: 744

calico-node:
  cmd.run:
    - name: calicoctl node
    - unless: docker ps | grep calico-node
    - env:
      - ETCD_AUTHORITY: "{{ grains.api_servers }}:6666"
    - require:
      - kmod: ip6_tables
      - kmod: xt_set
      - service: docker
      - file: calicoctl
      - cmd: etcd

etcd:
  cmd.run:
    - unless: docker ps | grep calico-etcd
    - name: >
               docker run --name calico-etcd -d --restart=always -p 6666:6666
               -v /varetcd:/var/etcd
               gcr.io/google_containers/etcd:2.0.8
               /usr/local/bin/etcd --name calico
               --data-dir /var/etcd/calico-data
               --advertise-client-urls http://{{grains.api_servers}}:6666
               --listen-client-urls http://0.0.0.0:6666
               --listen-peer-urls http://0.0.0.0:2380
               --initial-advertise-peer-urls http://{{grains.api_servers}}:2380
               --initial-cluster calico=http://{{grains.api_servers}}:2380

ip6_tables:
  kmod.present

xt_set:
  kmod.present

{% endif %}