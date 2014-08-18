include:
  - base

# The default GCE images have ip_forwarding explicitly set to 0.
# Here we take care of commenting that out.
net.ipv4.ip_forward:
  sysctl.present:
  - value: 1
  - config: /etc/sysctl.d/11-gce-network-security.conf

bridge-utils:
  pkg.installed

dummy0:
  network.managed:
    - enabled: True
    - type: eth
    - proto: static

cbr0:
  network.managed:
    - enabled: True
    - type: bridge
    - bridge: cbr0
    - delay: 0
    - proto: static
    - ipaddr: {{ grains['cbr-cidr'].split('/')[0] }}
    - netmask: {{ grains['cbr-cidr'].split('/')[1] }}
    - mtu: 1460
    - require:
      - pkg: bridge-utils
      - sysctl: net.ipv4.ip_forward
    - use:
      - network: dummy0
    - ports: dummy0
    - require:
      - network: dummy0


{% if grains['os_family'] == 'RedHat' %}

docker-io:
  pkg:
    - installed

/etc/sysconfig/docker:
  file.managed:
    - source: salt://docker/docker-defaults-fedora
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - makedirs: true
    - require:
      - network: cbr0

docker:
  service.running:
    - enable: True
    - watch:
      - pkg: docker-io
      - file: /etc/sysconfig/docker

{% else %}

docker-repo:
  pkgrepo.managed:
    - humanname: Docker Repo
    - name: deb https://get.docker.io/ubuntu docker main
    - key_url: https://get.docker.io/gpg
    - require:
      - pkg: pkg-core

lxc-docker:
  pkg.installed

/etc/default/docker:
  file.managed:
    - source: salt://docker/docker-defaults
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - makedirs: true
    - require:
      - network: cbr0

# With the correct dependencies, the race condition present here
# should no longer be a problem.
docker:
  service.running:
    - enable: True
    - require:
      - pkg: lxc-docker
      - network: cbr0
    - watch:
      - file: /etc/default/docker

{% endif %}
