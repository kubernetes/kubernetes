{% if grains['os_family'] == 'RedHat' %}
{% set environment_file = '/etc/sysconfig/docker' %}
{% else %}
{% set environment_file = '/etc/default/docker' %}
{% endif %}

bridge-utils:
  pkg.installed

{% if grains.os_family == 'RedHat' %}
docker-io:
  pkg:
    - installed

docker:
  service.running:
    - enable: True
    - require:
      - pkg: docker-io

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

# TODO: This should really be based on network strategy instead of os_family
net.ipv4.ip_forward:
  sysctl.present:
    - value: 1

cbr0:
  container_bridge.ensure:
    - cidr: {{ grains['cbr-cidr'] }}
    - mtu: 1460

purge-old-docker:
  pkg.removed:
    - pkgs:
      - lxc-docker-1.2.0
      - lxc-docker-1.3.0
      - lxc-docker-1.3.1
      - lxc-docker-1.3.2

{{ environment_file }}:
  file.managed:
    - source: salt://docker/docker-defaults
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - makedirs: true

# We are caching the Docker deb file in GCS for reliability and speed.  To
# update this to a new version of docker, do the following:
# 1. Find new deb name with:
#    curl https://get.docker.com/ubuntu/dists/docker/main/binary-amd64/Packages
# 2. Download based on that:
#    curl -O https://get.docker.com/ubuntu/pool/main/<...>
# 3. Upload to GCS:
#    gsutil cp <deb> gs://kubernetes-release/docker/<deb>
# 4. Make it world readable:
#    gsutil acl ch -R -g all:R gs://kubernetes-release/docker/<deb>
# 5. Get a hash of the deb:
#    shasum <deb>
# 6. Update this file with new deb name, new hash and new version
# 7. Add the old version to purge-old-docker above.

{% set storage_base='https://storage.googleapis.com/kubernetes-release/docker/' %}
{% set deb='lxc-docker-1.3.3_1.3.3_amd64.deb' %}
{% set deb_hash='sha1=7c0f4a8016dae234e66f13f922f24987db3d3ba4' %}
{% set docker_ver='1.3.3' %}

/var/cache/docker-install/{{ deb }}:
  file.managed:
    - source: {{ storage_base }}{{ deb }}
    - source_hash: {{ deb_hash }}
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

lxc-docker-{{ docker_ver }}:
  pkg.installed:
    - sources:
      - lxc-docker-{{ docker_ver }}: /var/cache/docker-install/{{ deb }}

docker:
  service.running:
    - enable: True
    - require:
      - pkg: lxc-docker-{{ docker_ver }}
    - watch:
      - file: {{ environment_file }}
      - container_bridge: cbr0
      - pkg: lxc-docker-{{ docker_ver }}

{% endif %}
