/etc/kubernetes/manifests/fluentd-gcp.yaml:
  file.managed:
    - source: salt://fluentd-gcp/fluentd-gcp.yaml
    - user: root
    - group: root
    - mode: 644
    - makedirs: true
    - dir_mode: 755
cron_restart_fluentd:
  cron.present:
    - name: docker restart `docker ps | grep fluent | grep -v '/pause' | grep 'Up' | awk '{ print $1 }'`:
    - user: root
    - minute: '*'
    - hour: '*/4'
    - daymonth: '*'
    - month: '*'
    - dayweek: '*'
