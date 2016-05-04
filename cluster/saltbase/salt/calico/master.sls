{% if pillar.get('policy_provider', '').lower() == 'calico' %}

calicoctl:
  file.managed:
    - name: /usr/bin/calicoctl
    - source: https://github.com/projectcalico/calico-docker/releases/download/v0.19.0/calicoctl
    - source_hash: sha256=6db00c94619e82d878d348c4e1791f8d2f0db59075f6c8e430fefae297c54d96
    - makedirs: True
    - mode: 744

calico-etcd:
  cmd.run:
    - unless: docker ps | grep calico-etcd
    - name: >
               docker run --name calico-etcd -d --restart=always -p 6666:6666
               -v /varetcd:/var/etcd
               gcr.io/google_containers/etcd:2.2.1
               /usr/local/bin/etcd --name calico
               --data-dir /var/etcd/calico-data
               --advertise-client-urls http://{{ grains.id }}:6666
               --listen-client-urls http://0.0.0.0:6666
               --listen-peer-urls http://0.0.0.0:6667
               --initial-advertise-peer-urls http://{{ grains.id }}:6667
               --initial-cluster calico=http://{{ grains.id }}:6667

calico-policy-agent:
  file.managed:
    - name: /etc/kubernetes/manifests/calico-policy-agent.manifest
    - source: salt://calico/calico-policy-agent.manifest
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - makedirs: true
    - dir_mode: 755
    - context:
        cpurequest: '20m'
    - require:
      - service: docker
      - service: kubelet
      - cmd: calico-etcd

{% endif -%}
