
# This runs highstate on the target node
highstate_run:
  cmd.state.highstate:
    - tgt: {{ data['id'] }}
