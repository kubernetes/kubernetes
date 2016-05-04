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
    - source: https://storage.googleapis.com/kubernetes-release/network-plugins/cni-c864f0e1ea73719b8f4582402b0847064f9883b0.tar.gz
    - tar_options: v
    - source_hash: md5=5f71ea8046930357e0ca83088064db93
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
