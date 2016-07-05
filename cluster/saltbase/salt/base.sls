pkg-core:
  pkg.installed:
    - names:
      - curl
{% if grains['os_family'] == 'RedHat' %}
      - python
      - git
      - socat
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

# Fix ARP cache issues on AWS by setting net.ipv4.neigh.default.gc_thresh1=0
# See issue #23395
{% if grains.get('cloud') == 'aws' %}
# Work around Salt #18089: https://github.com/saltstack/salt/issues/18089
# (we also have to give it a different id from the same fix elsewhere)
99-salt-conf-with-a-different-id:
  file.touch:
    - name: /etc/sysctl.d/99-salt.conf

net.ipv4.neigh.default.gc_thresh1:
  sysctl.present:
    - value: 0
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

/usr/local/share/doc/kubernetes/kubernetes-src.tar.gz:
  file.managed:
    - source: salt://kube-docs/kubernetes-src.tar.gz
    - user: root
    - group: root
    - mode: 644
