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
    - source: https://storage.googleapis.com/kubernetes-release/network-plugins/cni-42c4cb842dad606a84e93aad5a4484ded48e3046.tar.gz
    - tar_options: v
    - source_hash: md5=8cee1d59f01a27e8c2c10c120dce1e7d
    - archive_format: tar
    - if_missing: /opt/cni/bin

