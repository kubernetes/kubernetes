/etc/openvpn/server.conf:
  file.managed:
    - source: salt://openvpn/server.conf
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - makedirs: True

{% for (minion, grains) in salt['mine.get']('roles:kubernetes-pool', 'grains.items', expr_form='grain').items() %}
/etc/openvpn/ccd/{{ minion }}:
  file.managed:
    - contents: "iroute {{ grains['cbr-string'] }}\n"
    - user: root
    - group: root
    - mode: 644
    - makedirs: True
{% endfor %}

openssl dhparam -out /etc/openvpn/dh1024.pem 1024:
  cmd.run:
    - creates: /etc/openvpn/dh1024.pem
    - unless: file /etc/openvpn/dh1024.pem

openvpn:
  pkg:
    - latest
  service.running:
    - enable: True
    - watch:
      - file: /etc/openvpn/server.conf
