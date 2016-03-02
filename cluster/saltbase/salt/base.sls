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
      - nfs-common
      - socat
{% endif %}
# Ubuntu installs netcat-openbsd by default, but on GCE/Debian netcat-traditional is installed.
# They behave slightly differently.
# For sanity, we try to make sure we have the same netcat on all OSes (#15166)
{% if grains['os'] == 'Ubuntu' %}
      - netcat-traditional
{% endif %}
# Make sure git is installed for mounting git volumes
{% if grains['os'] == 'Ubuntu' %}
      - git
{% endif %}

/usr/local/share/doc/kubernetes:
  file.directory:
    - user: root
    - group: root
    - mode: 755
    - makedirs: True

/usr/local/share/doc/kubernetes/LICENSES:
  file.managed:
    - source: salt://kube-docs/LICENSES
    - user: root
    - group: root
    - mode: 644
