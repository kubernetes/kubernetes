/opt/cni:
  file.directory:
    - user: root
    - group: root
    - mode: 755
    - makedirs: True

/etc/cni/net.d:
  file.directory:
    - user: root
    - group: root
    - mode: 755
    - makedirs: True

# These are all available CNI network plugins.
cni-tar:
  archive:
    - extracted
    - user: root
    - name: /opt/cni
    - makedirs: True
    - source: https://storage.googleapis.com/kubernetes-release/network-plugins/cni-07a8a28637e97b22eb8dfe710eeae1344f69d16e.tar.gz
    - tar_options: v
    - source_hash: md5=dd11b04dcb0b9aca00b5287ba18dcfaa
    - archive_format: tar
    - if_missing: /opt/cni/bin

{% if grains['cloud'] is defined and grains.cloud in [ 'vagrant' ]  %}
# Install local CNI network plugins in a Vagrant environment
cmd-local-cni-plugins:
   cmd.run:
      - name: |
         cp -v /vagrant/cluster/network-plugins/cni/bin/* /opt/cni/bin/.
         chmod +x /opt/cni/bin/*
cmd-local-cni-config:
   cmd.run:
      - name: |
         cp -v /vagrant/cluster/network-plugins/cni/config/* /etc/cni/net.d/.
         chown root:root /etc/cni/net.d/*
         chmod 744 /etc/cni/net.d/*
{% endif -%}
