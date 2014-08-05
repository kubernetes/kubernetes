{% set go_version = '1.2.2' %}
{% set go_arch = 'linux-amd64' %}
{% set go_archive = 'go%s.%s.tar.gz' | format(go_version, go_arch) %}
{% set go_url = 'http://golang.org/dl/' + go_archive %}
{% set go_hash = 'sha1=6bd151ca49c435462c8bf019477a6244b958ebb5' %}

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
