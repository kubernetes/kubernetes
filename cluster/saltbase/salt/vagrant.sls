vagrant:
  user.present:
    - optional_groups:
        - docker
    - remove_groups: False
    - require:
      - pkg: docker-io
