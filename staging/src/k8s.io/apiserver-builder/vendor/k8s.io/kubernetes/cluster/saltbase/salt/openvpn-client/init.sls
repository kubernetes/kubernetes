/etc/openvpn/client.conf:
  file.managed:
    - source: salt://openvpn-client/client.conf
    - template: jinja
    - user: root
    - group: root
    - mode: 644
    - makedirs: True

openvpn:
  pkg:
    - latest
  service.running:
    - enable: True
    - watch:
      - file: /etc/openvpn/client.conf
