delete_cadvisor_manifest:
  file.absent:
    - name: /etc/kubernetes/manifests/cadvisor.manifest
