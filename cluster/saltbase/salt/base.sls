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
{% endif %}

{% if grains['oscodename'] == 'jessie' %}
is_systemd:
  grains.present:
    - value: True
systemd_system_path:
  grains.present:
    - value: /lib/systemd/system
{% elif grains['os_family'] == 'RedHat' %}
is_systemd:
  grains.present:
    - value: True
systemd_system_path:
  grains.present:
    - value: /usr/lib/systemd/system
{% else %}
is_systemd:
  grains.present:
    - value: False
{% endif %}
