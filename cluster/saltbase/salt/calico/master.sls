{% if pillar.get('network_provider', '').lower() == 'calico' %}

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
      - cmd: etcd

etcd:
  cmd.run:
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
