# Install all network plugins under /opt/cni, note this must match one of the directories
# the CNI plugin under kubelet/network/cni/ searches for.
/opt/cni:
  file.directory:
    - user: root
    - group: root
    - mode: 755
    - makedirs: True

# Network config for the one Kubernetes default network. Must match the
# --network-plugin-dir flag given to the Kubelet.
/etc/cni/net.d:
  file.directory:
    - user: root
    - group: root
    - mode: 755
    - makedirs: True

# These are all available CNI network plugins.
network-plugins-tar:
  archive:
    - extracted
    - user: root
    - name: /opt/cni
    - makedirs: True
    - source: https://storage.googleapis.com/kubernetes-release/network-plugins/cni-v0.1.0.tar.gz
    - tar_options: v
    - source_hash: md5=a21f366b13cd20da0809d9397f7890b5
    - archive_format: tar
    - if_missing: /opt/cni/bin

