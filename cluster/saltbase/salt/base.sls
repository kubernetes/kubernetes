pkg-core:
  pkg.installed:
    - names:
      - curl
{% if grains['os_family'] == 'RedHat' %}
      - python
      - git
      - glusterfs-fuse
{% else %}
      - apt-transport-https
      - python-apt
      - glusterfs-client
      - nfs-common
      - socat
{% endif %}
