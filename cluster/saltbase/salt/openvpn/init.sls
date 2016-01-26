/etc/openvpn/server.conf:
  file.managed:
    - source: salt://openvpn/server.conf
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - makedirs: True

{% for minion in salt['mine.get']('roles:kubernetes-pool', 'grains.items', expr_form='grain').values() %}
/etc/openvpn/ccd/{{ minion['hostnamef'] }}:
  file.managed:
    - contents: "iroute {{ minion['cbr-string'] }}\n"
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
