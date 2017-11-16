{% if grains['cloud'] is defined and grains.cloud in ['aws', 'gce', 'vagrant', 'photon-controller', 'openstack'] %}
# TODO: generate and distribute tokens on other cloud providers.
/srv/kubernetes/known_tokens.csv:
  file.managed:
    - source: salt://kube-apiserver/known_tokens.csv
    - user: root
    - group: root
    - mode: 600
#    - watch_in:
#      - service: kube-apiserver

/srv/kubernetes/basic_auth.csv:
  file.managed:
    - source: salt://kube-apiserver/basic_auth.csv
    - user: root
    - group: root
    - mode: 600

/srv/kubernetes/abac-authz-policy.jsonl:
  file.managed:
    - source: salt://kube-apiserver/abac-authz-policy.jsonl
    - template: jinja
    - user: root
    - group: root
    - mode: 600
{% endif %}

/var/log/kube-apiserver.log:
  file.managed:
    - user: root
    - group: root
    - mode: 644

/var/log/kube-apiserver-audit.log:
  file.managed:
    - user: root
    - group: root
    - mode: 644

# Copy kube-apiserver manifest to manifests folder for kubelet.
# Current containervm image by default has both docker and kubelet
# running. But during cluster creation stage, docker and kubelet
# could be overwritten completely, or restarted due to flag changes.
# The ordering of salt states for service docker, kubelet and
# master-addon below is very important to avoid the race between
# salt restart docker or kubelet and kubelet start master components.
# Without the ordering of salt states, when gce instance boot up,
# configure-vm.sh will run and download the release. At the end of
# boot, run-salt will installs kube-apiserver.manifest files to
# kubelet config directory before the installation of proper version
# kubelet. Please see
# http://issue.k8s.io/10122#issuecomment-114566063
# for detail explanation on this very issue.
/etc/kubernetes/manifests/kube-apiserver.manifest:
  file.managed:
    - source: salt://kube-apiserver/kube-apiserver.manifest
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - makedirs: true
    - dir_mode: 755
    - require:
      - service: docker
      - service: kubelet

#stop legacy kube-apiserver service
stop_kube-apiserver:
  service.dead:
    - name: kube-apiserver
    - enable: None
