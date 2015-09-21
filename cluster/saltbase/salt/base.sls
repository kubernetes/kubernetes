pkg-core:
  pkg.installed:
    - names:
      - curl
{% if grains['os_family'] == 'RedHat' %}
      - python
      - git
{% else %}
      - apt-transport-https
      - python-apt
{% endif %}