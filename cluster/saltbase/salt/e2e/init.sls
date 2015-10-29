e2e:

{% if grains['os_family'] == 'Debian' and  grains['oscodename'] == 'wheezy' %}
  # Add GlusterFS 3.5.2 Debian repo - Wheezy has Gluster 3.2 and that's too old for us.
  pkgrepo.managed:
    - name: deb http://download.gluster.org/pub/gluster/glusterfs/3.5/3.5.2/Debian/wheezy/apt wheezy main
    - dist: wheezy
    - file: /etc/apt/sources.list.d/gluster.list
    - key_url: http://download.gluster.org/pub/gluster/glusterfs/3.5/3.5.2/Debian/wheezy/pubkey.gpg
{% endif %}

  # Install various packages required by e2e tests to all hosts.
  # 'pkg.latest' is used to install updated glusterfs-client from the repo above,
  # GCE image already has glusterfs-client 3.2.5, which is too old.
  pkg.latest:
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
