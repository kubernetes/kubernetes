/opt/cni:
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
    - source: https://storage.googleapis.com/kubernetes-release/network-plugins/cni-09214926.tar.gz
    - tar_options: v
    - source_hash: md5=58f8631f912dd88be6a0920775db7274
    - archive_format: tar
    - if_missing: /opt/cni/bin

