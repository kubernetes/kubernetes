/etc/cron.d/dhcp:
  file:
    - managed
    - source: salt://dhcp-cron/cron
    - user: root
    - group: root
    - mode: 644
