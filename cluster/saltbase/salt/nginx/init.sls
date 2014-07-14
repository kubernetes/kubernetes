nginx:
  pkg:
    - installed
  service:
    - running
    - watch:
      - pkg: nginx
      - file: /etc/nginx/nginx.conf
      - file: /etc/nginx/sites-enabled/default
      - file: /usr/share/nginx/htpasswd
      - cmd: /usr/share/nginx/server.cert

/usr/share/nginx/server.cert:
  cmd.script:
    - source: salt://nginx/make-cert.sh
    - cwd: /
    - user: root
    - group: root
    - shell: /bin/bash
    - stateful: True

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

/usr/share/nginx/make-cert.sh:
  file:
    - managed
    - source: salt://nginx/make-cert.sh
    - user: root
    - group: root
    - mode: 755
