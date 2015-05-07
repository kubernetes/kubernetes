{% if grains['os_family'] != 'RedHat' %}

atop:
  pkg:
    - installed

atop-service:
  service:
    - running
    - name: atop
    - watch:
      - pkg: atop
      - file: /etc/init.d/atop

{% endif %}
