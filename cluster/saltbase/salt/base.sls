pkg-core:
  pkg.latest:
    - names:
{% if grains['os_family'] == 'RedHat' %}
      - python
      - git
{% else %}
      - apt-transport-https
      - python-apt
{% endif %}