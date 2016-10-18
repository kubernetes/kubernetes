{% if grains['oscodename'] in [ 'vivid', 'wily', 'jessie', 'xenial', 'yakkety' ] %}
is_systemd: True
systemd_system_path: /lib/systemd/system
{% elif grains['os_family'] == 'RedHat' %}
is_systemd: True
systemd_system_path: /usr/lib/systemd/system
{% else %}
is_systemd: False
{% endif %}
