# Add static routes to every minion to enable pods in the 10.244.x.x range to
# reach each other. This is suboptimal, but necessary to let every pod have
# its IP and have pods between minions be able to talk with each other.
# This will be obsolete when we figure out the right way to make this work.

/etc/network/if-up.d/static-routes:
  file.managed:
    - source: salt://static-routes/if-up
    - template: jinja
    - user: root
    - group: root
    - mode: 755

/etc/network/if-down.d/static-routes:
  file.managed:
    - source: salt://static-routes/if-down
    - template: jinja
    - user: root
    - group: root
    - mode: 755

refresh-routes:
  cmd.wait_script:
    - source: salt://static-routes/refresh
    - cwd: /etc/network/
    - user: root
    - group: root
    - watch:
      - file: /etc/network/if-up.d/static-routes
      - file: /etc/network/if-down.d/static-routes
