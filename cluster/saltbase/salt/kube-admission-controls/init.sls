{% if 'LimitRanger' in pillar.get('admission_control', '') %}
/etc/kubernetes/admission-controls/limit-range:
  file.recurse:
    - source: salt://kube-admission-controls/limit-range
    - include_pat: E@(^.+\.yaml$|^.+\.json$)
    - user: root
    - group: root
    - dir_mode: 755
    - file_mode: 644
{% endif %}
