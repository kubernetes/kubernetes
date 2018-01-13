/home/kubernetes:
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
    - name: /home/kubernetes/bin
    - makedirs: True
    - source: https://storage.googleapis.com/kubernetes-release/network-plugins/cni-plugins-amd64-v0.6.0.tgz
    - tar_options: v
    - source_hash: md5=9534876FAE7DBE813CDAB404DC1F9219
    - archive_format: tar
    - if_missing: /home/kubernetes/bin
