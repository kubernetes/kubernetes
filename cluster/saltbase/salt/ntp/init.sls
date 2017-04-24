ntp:
  pkg:
    - installed

ntp-service:
  service:
    - running
{% if grains['os_family'] == 'RedHat' %}
    - name: ntpd
{% else %}
    - name: ntp
{% endif %}
    - watch:
      - pkg: ntp

