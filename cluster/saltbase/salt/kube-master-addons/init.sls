/etc/kubernetes/kube-master-addons.sh:
  file.managed:
    - source: salt://kube-master-addons/kube-master-addons.sh
    - user: root
    - group: root
    - mode: 755

{% if grains['os_family'] == 'RedHat' %}

/usr/lib/systemd/system/kube-master-addons.service:
  file.managed:
    - source: salt://kube-master-addons/kube-master-addons.service
    - user: root
    - group: root

{% else %}

/etc/init.d/kube-master-addons:
  file.managed:
    - source: salt://kube-master-addons/initd
    - user: root
    - group: root
    - mode: 755

{% endif %}

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

kube-master-addons:
  service.running:
    - enable: True
    - restart: True
    - watch:
      - file: master-docker-image-tags
      - file: /etc/kubernetes/kube-master-addons.sh
