# This runs highstate only on the NEW node, regardless of type.
highstate_new:
  cmd.state.highstate:
    - tgt: {{ data['id'] }}
