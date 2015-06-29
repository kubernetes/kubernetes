/etc/kubernetes/kube-master-addons.sh:
  file.managed:
    - source: salt://kube-master-addons/kube-master-addons.sh
    - user: root
    - group: root
    - mode: 755

# Used to restart kube-master-addons service each time salt is run
# Actually, it doens't work (the service is not restarted),
# but master-addon service always terminates after it does it job,
# so it is (usually) not running and it will be started when
# salt is run.
# This salt state is not removed because there is a risk
# of introducing regression in 1.0. Please remove it afterwards.
# See also the salt config for kube-addons to see how to restart
# a service on demand.
master-docker-image-tags:
  file.touch:
    - name: /srv/pillar/docker-images.sls

{% if pillar.get('is_systemd') %}

{{ pillar.get('systemd_system_path') }}/kube-master-addons.service:
  file.managed:
    - source: salt://kube-master-addons/kube-master-addons.service
    - user: root
    - group: root
  cmd.wait:
    - name: /opt/kubernetes/helpers/services bounce kube-master-addons
    - watch:
      - file: master-docker-image-tags
      - file: /etc/kubernetes/kube-master-addons.sh
      - file: {{ pillar.get('systemd_system_path') }}/kube-master-addons.service

{% else %}

/etc/init.d/kube-master-addons:
  file.managed:
    - source: salt://kube-master-addons/initd
    - user: root
    - group: root
    - mode: 755

# Current containervm image by default has both docker and kubelet
# running. But during cluster creation stage, docker and kubelet
# could be overwritten completely, or restarted due to flag changes.
# The ordering of salt states for service docker, kubelet and
# master-addon below is very important to avoid the race between
# salt restart docker or kubelet and kubelet start master components.
# Without the ordering of salt states, when gce instance boot up,
# configure-vm.sh will run and download the release. At the end of
# boot, run-salt will run kube-master-addons service which installs
# master component manifest files to kubelet config directory before
# the installation of proper version kubelet. Please see
# https://github.com/GoogleCloudPlatform/kubernetes/issues/10122#issuecomment-114566063
# for detail explanation on this very issue.
kube-master-addons:
  service.running:
    - enable: True
    - restart: True
    - watch:
      - file: master-docker-image-tags
      - file: /etc/kubernetes/kube-master-addons.sh

{% endif %}
