# This runs highstate on the minion nodes.
#
# Some of the cluster deployment scripts use the list of minions on the minions
# themselves. To propagate changes throughout
# the pool, run highstate on all minions whenever a single minion starts.
#
highstate_minions:
  cmd.state.highstate:
    - tgt: 'roles:kubernetes-pool'
    - expr_form: grain
