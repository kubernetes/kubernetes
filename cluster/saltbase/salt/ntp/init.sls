ntp:
  pkg:
    - installed

ntp-service:
  service:
    - running
    - name: ntp
    - watch:
      - pkg: ntp

