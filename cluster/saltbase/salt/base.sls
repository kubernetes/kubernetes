pkg-core:
  pkg.installed:
    - names:
{% if grains['os_family'] == 'RedHat' %}
      - python
      - git
{% else %}
      - apt-transport-https
      - python-apt
{% endif %}