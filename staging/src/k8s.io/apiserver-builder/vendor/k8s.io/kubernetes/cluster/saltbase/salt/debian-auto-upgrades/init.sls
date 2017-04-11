{% if grains['os_family'] == 'Debian' %}
unattended-upgrades:
  pkg.installed

'/etc/apt/apt.conf.d/20auto-upgrades':
  file.managed:
    - source: salt://debian-auto-upgrades/20auto-upgrades
    - user: root
    - group: root
    - mode: 644
    - require:
      - pkg: unattended-upgrades
{% endif %}
