e2e:
  # Install various packages required by e2e tests to all hosts.
  pkg.installed:
    - refresh: true
    - pkgs:
      - targetcli
      - ceph
{% if grains['os_family'] == 'RedHat' %}
      - glusterfs-fuse
      - rbd-fuse
      - iscsi-initiator-utils
      - nfs-utils
{% else %}
      - glusterfs-client
      - open-iscsi
      - iscsitarget-dkms
      - nfs-common
{% endif %}



{% if grains['os_family'] == 'Debian' %}
# On Debian, re-start open-iscsi to generate unique
# /etc/iscsi/initiatorname.iscsi
open-iscsi:
  cmd.run:
    - name: 'service open-iscsi restart'
{% endif %}
