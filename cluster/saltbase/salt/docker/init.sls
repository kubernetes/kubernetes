{% if pillar.get('is_systemd') %}
  {% set environment_file = '/etc/sysconfig/docker' %}
{% else %}
  {% set environment_file = '/etc/default/docker' %}
{% endif %}

bridge-utils:
  pkg.installed

{% if grains.os_family == 'RedHat' %}

{{ environment_file }}:
  file.managed:
    - source: salt://docker/default
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - makedirs: true

{% if (grains.os == 'Fedora' and grains.osrelease_info[0] >= 22) or (grains.os == 'CentOS' and grains.osrelease_info[0] >= 7) %}

docker:
  pkg:
    - installed
  service.running:
    - enable: True
    - require:
      - pkg: docker
    - watch:
      - file: {{ environment_file }}
      - pkg: docker

{% else %}

docker-io:
  pkg:
    - installed

docker:
  service.running:
    - enable: True
    - require:
      - pkg: docker-io
    - watch:
      - file: {{ environment_file }}
      - pkg: docker-io

{% endif %}
{% elif grains.cloud is defined and grains.cloud == 'azure-legacy' %}

{% if pillar.get('is_systemd') %}

{{ pillar.get('systemd_system_path') }}/docker.service:
  file.managed:
    - source: salt://docker/docker.service
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - defaults:
        environment_file: {{ environment_file }}

# The docker service.running block below doesn't work reliably
# Instead we run our script which e.g. does a systemd daemon-reload
# But we keep the service block below, so it can be used by dependencies
# TODO: Fix this
fix-service-docker:
  cmd.wait:
    - name: /opt/kubernetes/helpers/services bounce docker
    - watch:
      - file: {{ pillar.get('systemd_system_path') }}/docker.service
      - file: {{ environment_file }}
{% endif %}

{{ environment_file }}:
  file.managed:
    - source: salt://docker/docker-defaults
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - makedirs: true
    - require:
      - pkg: docker-engine

apt-key:
  pkgrepo.managed:
    - humanname: Dotdeb
    - name: deb https://apt.dockerproject.org/repo ubuntu-trusty main
    - dist: ubuntu-trusty
    - file: /etc/apt/sources.list.d/docker.list
    - keyid: 58118E89F3A912897C070ADBF76221572C52609D
    - keyserver: hkp://p80.pool.sks-keyservers.net:80

lxc-docker:
  pkg:
    - purged

docker-io:
  pkg:
    - purged

cbr0:
  network.managed:
    - enabled: True
    - type: bridge
{% if grains['roles'][0] == 'kubernetes-pool' %}
    - proto: none
{% else %}
    - proto: dhcp
{% endif %}
    - ports: none
    - bridge: cbr0
{% if grains['roles'][0] == 'kubernetes-pool' %}
    - ipaddr: {{ grains['cbr-cidr'] }}
{% endif %}
    - delay: 0
    - bypassfirewall: True
    - require_in:
      - service: docker

docker-engine:
   pkg:
     - installed
     - require:
       - pkgrepo: 'apt-key'

docker:
   service.running:
     - enable: True
     - require:
       - file: {{ environment_file }}
     - watch:
       - file: {{ environment_file }}

{% elif grains.cloud is defined and grains.cloud in ['vsphere', 'photon-controller'] and grains.os == 'Debian' and grains.osrelease_info[0] >=8 %}

{% if pillar.get('is_systemd') %}

/opt/kubernetes/helpers/docker-prestart:
  file.managed:
    - source: salt://docker/docker-prestart
    - user: root
    - group: root
    - mode: 755

{{ pillar.get('systemd_system_path') }}/docker.service:
  file.managed:
    - source: salt://docker/docker.service
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - defaults:
        environment_file: {{ environment_file }}
    - require:
      - file: /opt/kubernetes/helpers/docker-prestart
      - pkg: docker-engine

# The docker service.running block below doesn't work reliably
# Instead we run our script which e.g. does a systemd daemon-reload
# But we keep the service block below, so it can be used by dependencies
# TODO: Fix this
fix-service-docker:
  cmd.wait:
    - name: /opt/kubernetes/helpers/services bounce docker
    - watch:
      - file: {{ pillar.get('systemd_system_path') }}/docker.service
      - file: {{ environment_file }}
{% endif %}

{{ environment_file }}:
  file.managed:
    - source: salt://docker/docker-defaults
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - makedirs: true
    - require:
      - pkg: docker-engine

apt-key:
   cmd.run:
     - name: 'apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys 58118E89F3A912897C070ADBF76221572C52609D'
     - unless: 'apt-key finger | grep "5811 8E89"'

apt-update:
  cmd.run:
    - name: '/usr/bin/apt-get update -y'
    - require:
       - cmd : 'apt-key'

lxc-docker:
  pkg:
    - purged

docker-io:
  pkg:
    - purged

cbr0:
  network.managed:
    - enabled: True
    - type: bridge
    - proto: dhcp
    - ports: none
    - bridge: cbr0
    - delay: 0
    - bypassfirewall: True
    - require_in:
      - service: docker

/etc/apt/sources.list.d/docker.list:
  file.managed:
    - source: salt://docker/docker.list
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - require:
      - cmd: 'apt-update'

# restricting docker version to 1.9. with older version of docker we are facing
# issue https://github.com/docker/docker/issues/18793.
# newer version of docker 1.10.0 is not well tested yet.
# full comments: https://github.com/kubernetes/kubernetes/pull/20851
docker-engine:
   pkg:
     - installed
     - version: 1.9.*
     - require:
       - file: /etc/apt/sources.list.d/docker.list
docker:
   service.running:
     - enable: True
     - require:
       - file: {{ environment_file }}
     - watch:
       - file: {{ environment_file }}

{% else %}

{% if grains.cloud is defined
   and grains.cloud == 'gce' %}
# The default GCE images have ip_forwarding explicitly set to 0.
# Here we take care of commenting that out.
/etc/sysctl.d/11-gce-network-security.conf:
  file.replace:
    - pattern: '^net.ipv4.ip_forward=0'
    - repl: '# net.ipv4.ip_forward=0'
{% endif %}

# Work around Salt #18089: https://github.com/saltstack/salt/issues/18089
/etc/sysctl.d/99-salt.conf:
  file.touch

# TODO: This should really be based on network strategy instead of os_family
net.ipv4.ip_forward:
  sysctl.present:
    - value: 1

{% if pillar.get('softlockup_panic', '').lower() == 'true' %}
# TODO(dchen1107) Remove this once kernel.softlockup_panic is built into the CVM image.
/etc/sysctl.conf:
  file.append:
    - text:
      - "kernel.softlockup_panic = 1"
      - "kernel.softlockup_all_cpu_backtrace = 1"

'sysctl-reload':
  cmd.run:
    - name: 'sysctl --system'
    - unless: 'sysctl -a | grep "kernel.softlockup_panic = 1"'
{% endif %}
 
{{ environment_file }}:
  file.managed:
    - source: salt://docker/docker-defaults
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - makedirs: true

# Docker is on the ContainerVM image by default. The following
# variables are provided for other cloud providers, and for testing and dire circumstances, to allow
# overriding the Docker version that's in a ContainerVM image.
#
# To change:
#
# 1. Find new deb name at:
#    http://apt.dockerproject.org/repo/pool/main/d/docker-engine
# 2. Download based on that:
#    curl -O http://apt.dockerproject.org/repo/pool/main/d/docker-engine/<deb>
# 3. Upload to GCS:
#    gsutil cp <deb> gs://kubernetes-release/docker/<deb>
# 4. Make it world readable:
#    gsutil acl ch -R -g all:R gs://kubernetes-release/docker/<deb>
# 5. Get a hash of the deb:
#    shasum <deb>
# 6. Update override_deb, override_deb_sha1, override_docker_ver with new
#    deb name, new hash and new version

{% set storage_base='https://storage.googleapis.com/kubernetes-release/docker/' %}

{% set override_deb_url='' %}

{% if grains.get('cloud', '') == 'gce'
   and grains.get('os_family', '') == 'Debian'
   and grains.get('oscodename', '') == 'wheezy' -%}
{% set docker_pkg_name='' %}
{% set override_deb='' %}
{% set override_deb_sha1='' %}
{% set override_docker_ver='' %}

{% elif grains.get('cloud', '') == 'gce'
   and grains.get('os_family', '') == 'Debian'
   and grains.get('oscodename', '') == 'jessie' -%}
{% set docker_pkg_name='' %}
{% set override_deb='' %}
{% set override_deb_sha1='' %}
{% set override_docker_ver='' %}

{% elif grains.get('cloud', '') == 'aws'
   and grains.get('os_family', '') == 'Debian'
   and grains.get('oscodename', '') == 'jessie' -%}
# TODO: Get from google storage?
{% set docker_pkg_name='docker-engine' %}
{% set override_docker_ver='1.11.2-0~jessie' %}
{% set override_deb='docker-engine_1.11.2-0~jessie_amd64.deb' %}
{% set override_deb_url='http://apt.dockerproject.org/repo/pool/main/d/docker-engine/docker-engine_1.11.2-0~jessie_amd64.deb' %}
{% set override_deb_sha1='c312f1f6fa0b34df4589bb812e4f7af8e28fd51d' %}

# Ubuntu presents as os_family=Debian, osfullname=Ubuntu
{% elif grains.get('cloud', '') == 'aws'
   and grains.get('os_family', '') == 'Debian'
   and grains.get('oscodename', '') == 'trusty' -%}
# TODO: Get from google storage?
{% set docker_pkg_name='docker-engine' %}
{% set override_docker_ver='1.11.2-0~trusty' %}
{% set override_deb='docker-engine_1.11.2-0~trusty_amd64.deb' %}
{% set override_deb_url='http://apt.dockerproject.org/repo/pool/main/d/docker-engine/docker-engine_1.11.2-0~trusty_amd64.deb' %}
{% set override_deb_sha1='022dee31e68c6d572eaac750915786e4a6729d2a' %}

{% elif grains.get('cloud', '') == 'aws'
   and grains.get('os_family', '') == 'Debian'
   and grains.get('oscodename', '') == 'wily' -%}
# TODO: Get from google storage?
{% set docker_pkg_name='docker-engine' %}
{% set override_docker_ver='1.11.2-0~wily' %}
{% set override_deb='docker-engine_1.11.2-0~wily_amd64.deb' %}
{% set override_deb_url='http://apt.dockerproject.org/repo/pool/main/d/docker-engine/docker-engine_1.11.2-0~wily_amd64.deb' %}
{% set override_deb_sha1='3e02f51fe18aa777eeb1676c3d9a75e5ea6d96c9' %}

{% else %}
{% set docker_pkg_name='lxc-docker-1.7.1' %}
{% set override_docker_ver='1.7.1' %}
{% set override_deb='lxc-docker-1.7.1_1.7.1_amd64.deb' %}
{% set override_deb_sha1='81abef31dd2c616883a61f85bfb294d743b1c889' %}
{% endif %}

{% if override_deb_url == '' %}
{% set override_deb_url=storage_base + override_deb %}
{% endif %}

{% if override_docker_ver != '' %}
purge-old-docker-package:
  pkg.removed:
    - pkgs:
      - lxc-docker-1.6.2

/var/cache/docker-install/{{ override_deb }}:
  file.managed:
    - source: {{ override_deb_url }}
    - source_hash: sha1={{ override_deb_sha1 }}
    - user: root
    - group: root
    - mode: 644
    - makedirs: true

# Drop the license file into /usr/share so that everything is crystal clear.
/usr/share/doc/docker/apache.txt:
  file.managed:
    - source: {{ storage_base }}apache2.txt
    - source_hash: sha1=2b8b815229aa8a61e483fb4ba0588b8b6c491890
    - user: root
    - group: root
    - mode: 644
    - makedirs: true

libltdl7:
  pkg.installed

docker-upgrade:
  cmd.run:
    - name: /opt/kubernetes/helpers/pkg install-no-start {{ docker_pkg_name }} {{ override_docker_ver }} /var/cache/docker-install/{{ override_deb }}
    - require:
      - file: /var/cache/docker-install/{{ override_deb }}
      - pkg: libltdl7

{% endif %} # end override_docker_ver != ''

{% if pillar.get('is_systemd') %}

/opt/kubernetes/helpers/docker-prestart:
  file.managed:
    - source: salt://docker/docker-prestart
    - user: root
    - group: root
    - mode: 755

# Default docker systemd unit file doesn't use an EnvironmentFile; replace it with one that does.
{{ pillar.get('systemd_system_path') }}/docker.service:
  file.managed:
    - source: salt://docker/docker.service
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - defaults:
        environment_file: {{ environment_file }}
    - require:
      - file: /opt/kubernetes/helpers/docker-prestart

# The docker service.running block below doesn't work reliably
# Instead we run our script which e.g. does a systemd daemon-reload
# But we keep the service block below, so it can be used by dependencies
# TODO: Fix this
fix-service-docker:
  cmd.wait:
    - name: /opt/kubernetes/helpers/services enable docker
    - watch:
      - file: {{ pillar.get('systemd_system_path') }}/docker.service
      - file: {{ environment_file }}
{% if override_docker_ver != '' %}
    - require:
      - cmd: docker-upgrade
{% endif %}

/opt/kubernetes/helpers/docker-healthcheck:
  file.managed:
    - source: salt://docker/docker-healthcheck
    - user: root
    - group: root
    - mode: 755

{{ pillar.get('systemd_system_path') }}/docker-healthcheck.service:
  file.managed:
    - source: salt://docker/docker-healthcheck.service
    - template: jinja
    - user: root
    - group: root
    - mode: 644

{{ pillar.get('systemd_system_path') }}/docker-healthcheck.timer:
  file.managed:
    - source: salt://docker/docker-healthcheck.timer
    - template: jinja
    - user: root
    - group: root
    - mode: 644

# Tell systemd to load the timer
fix-systemd-docker-healthcheck-timer:
  cmd.wait:
    - name: /opt/kubernetes/helpers/services bounce docker-healthcheck.timer
    - watch:
      - file: {{ pillar.get('systemd_system_path') }}/docker-healthcheck.timer

# Trigger a first run of docker-healthcheck; needed because the timer fires 10s after the previous run.
fix-systemd-docker-healthcheck-service:
  cmd.wait:
    - name: /opt/kubernetes/helpers/services bounce docker-healthcheck.service
    - watch:
      - file: {{ pillar.get('systemd_system_path') }}/docker-healthcheck.service
    - require:
      - cmd: fix-service-docker

{% endif %}

docker:
# Starting Docker is racy on aws for some reason.  To be honest, since Monit
# is managing Docker restart we should probably just delete this whole thing
# but the kubernetes components use salt 'require' to set up a dag, and that
# complicated and scary to unwind.
# On AWS, we use a trick now... We don't start the docker service through Salt.
# Kubelet or our health checker will start it.  But we use service.enabled,
# so we still have a `service: docker` node for our DAG.
{% if grains.cloud is defined and grains.cloud == 'aws' %}
  service.enabled:
{% else %}
  service.running:
    - enable: True
{% endif %}
# If we put a watch on this, salt will try to start the service.
# We put the watch on the fixer instead
{% if not pillar.get('is_systemd') %}
    - watch:
      - file: {{ environment_file }}
{% if override_docker_ver != '' %}
      - cmd: docker-upgrade
{% endif %}
{% endif %}
    - require:
      - file: {{ environment_file }}
{% if override_docker_ver != '' %}
      - cmd: docker-upgrade
{% endif %}
{% if pillar.get('is_systemd') %}
      - cmd: fix-service-docker
{% endif %}
{% endif %} # end grains.os_family != 'RedHat'

