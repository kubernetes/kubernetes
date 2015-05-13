{% if grains['os_family'] == 'RedHat' %}
{% set environment_file = '/etc/sysconfig/kubelet' %}
{% else %}
{% set environment_file = '/etc/default/kubelet' %}
{% endif %}

{{ environment_file}}:
  file.managed:
    - source: salt://kubelet/default
    - template: jinja
    - user: root
    - group: root
    - mode: 644

/usr/local/bin/kubelet:
  file.managed:
    - source: salt://kube-bins/kubelet
    - user: root
    - group: root
    - mode: 755

{% if grains['os_family'] == 'RedHat' %}

/usr/lib/systemd/system/kubelet.service:
  file.managed:
    - source: salt://kubelet/kubelet.service
    - user: root
    - group: root

{% else %}

/etc/init.d/kubelet:
  file.managed:
    - source: salt://kubelet/initd
    - user: root
    - group: root
    - mode: 755

{% endif %}

# The default here is that this file is blank.  If this is the case, the kubelet
# won't be able to parse it as JSON and will try to use the kubernetes_auth file
# instead.  You'll see a single error line in the kubelet start up file
# about this.
/var/lib/kubelet/kubeconfig:
  file.managed:
    - source: salt://kubelet/kubeconfig
    - user: root
    - group: root
    - mode: 400
    - makedirs: true

#
# --- This file is DEPRECATED ---
# The default here is that this file is blank.  If this is the case, the kubelet
# won't be able to parse it as JSON and it'll not be able to publish events to
# the apiserver.  You'll see a single error line in the kubelet start up file
# about this.
/var/lib/kubelet/kubernetes_auth:
  file.managed:
    - source: salt://kubelet/kubernetes_auth
    - user: root
    - group: root
    - mode: 400
    - makedirs: true

kubelet:
  service.running:
    - enable: True
    - watch:
      - file: /usr/local/bin/kubelet
{% if grains['os_family'] != 'RedHat' %}
      - file: /etc/init.d/kubelet
{% endif %}
      - file: /var/lib/kubelet/kubernetes_auth
