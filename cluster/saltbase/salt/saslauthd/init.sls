cyrus-sasl:
  pkg:
    - installed

saslauthd:
  service.running:
    - enable: true
{% if grains['os_family'] != 'RedHat' %}
    - watch:
      - file: /etc/init.d/saslauthd
{% endif %}
