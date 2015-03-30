delete_cadvisor_manifest:
  file.absent:
    - name: /etc/lmktfyrnetes/manifests/cadvisor.manifest
