nginx:
  pkg:
    - installed

/etc/nginx/nginx.conf:
  file:
    - managed
    - source: salt://nginx/nginx.conf
    - template: jinja
    - user: root
    - group: root
    - mode: 644

/etc/nginx/sites-enabled/default:
  file:
    - managed
    - makedirs: true
    - source: salt://nginx/kubernetes-site
    - user: root
    - group: root
    - mode: 644

/usr/share/nginx/htpasswd:
  file:
    - managed
    - source: salt://nginx/htpasswd
    - user: root
    - group: root
    - mode: 644

{% if grains.cloud is defined and grains.cloud in ['gce'] %}
/etc/kubernetes/manifests/nginx.json:
  file:
    - managed
    - source: salt://nginx/nginx.json
    - user: root
    - group: root
    - mode: 644
    - require:
      - file: /etc/nginx/nginx.conf
      - file: /etc/nginx/sites-enabled/default
      - file: /usr/share/nginx/htpasswd
      - cmd: kubernetes-cert      


#stop legacy nginx_service 
stop_nginx-service:
  service.dead:
    - name: nginx
    - enable: None

{% else %}
nginx-service:
  service:
    - running
    - name: nginx
    - watch:
      - pkg: nginx
      - file: /etc/nginx/nginx.conf
      - file: /etc/nginx/sites-enabled/default
      - file: /usr/share/nginx/htpasswd
      - cmd: kubernetes-cert
{% endif %}

