# This runs highstate on the nodes.
#
# Some of the cluster deployment scripts use the list of nodes on the nodes
# themselves (for example: every node is configured with static routes to
# every other node on a vSphere deployment). To propagate changes throughout
# the pool, run highstate on all nodes whenever a single node starts.
#
highstate_nodes:
  cmd.state.highstate:
    - tgt: 'roles:kubernetes-pool'
    - expr_form: grain
