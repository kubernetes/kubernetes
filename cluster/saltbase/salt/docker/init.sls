{% if grains['os_family'] == 'RedHat' %}
{% set environment_file = '/etc/sysconfig/docker' %}
{% else %}
{% set environment_file = '/etc/default/docker' %}
{% endif %}

{% if grains['os_family'] != 'RedHat' %}

docker-repo:
  pkgrepo.managed:
    - humanname: Docker Repo
    - name: deb https://get.docker.io/ubuntu docker main
    - key_url: https://get.docker.io/gpg
    - require:
      - pkg: pkg-core

# The default GCE images have ip_forwarding explicitly set to 0.
# Here we take care of commenting that out.
/etc/sysctl.d/11-gce-network-security.conf:
  file.replace:
    - pattern: '^net.ipv4.ip_forward=0'
    - repl: '# net.ipv4.ip_forward=0'

net.ipv4.ip_forward:
  sysctl.present:
    - value: 1

bridge-utils:
  pkg.latest

cbr0:
  container_bridge.ensure:
    - cidr: {{ grains['cbr-cidr'] }}
    - mtu: 1460

{% endif %}

{% if grains['os_family'] == 'RedHat' %}

docker-io:
  pkg:
    - installed

docker:
  service.running:
    - enable: True
    - require: 
      - pkg: docker-io

{% else %}

{{ environment_file }}:
  file.managed:
    - source: salt://docker/docker-defaults
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - makedirs: true

lxc-docker:
  pkg.latest

# There is a race here, I think.  As the package is installed, it will start
# docker.  If it doesn't write its pid file fast enough then this next stanza
# will try to ensure that docker is running.  That might start another copy of
# docker causing the thing to get wedged.
#
# See docker issue https://github.com/dotcloud/docker/issues/6184

# docker:
#   service.running:
#     - enable: True
#     - require:
#       - pkg: lxc-docker
#     - watch:
#       - file: /etc/default/docker

{% endif %}
