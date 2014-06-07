{% set go_version = '1.2' %}
{% set go_arch = 'linux-amd64' %}
{% set go_archive = 'go%s.%s.tar.gz' | format(go_version, go_arch) %}
{% set go_url = 'https://go.googlecode.com/files/' + go_archive %}
{% set go_hash = 'md5=68901bbf8a04e71e0b30aa19c3946b21' %}


get-golang:
  file.managed:
    - name: /var/cache/{{ go_archive }}
    - source: {{ go_url }}
    - source_hash: {{ go_hash }}
  cmd.wait:
    - cwd: /usr/local
    - name: tar xzf /var/cache/{{ go_archive }}
    - watch:
      - file: get-golang

install-golang:
  file.symlink:
    - name: /usr/local/bin/go
    - target: /usr/local/go/bin/go
    - watch:
      - cmd: get-golang
