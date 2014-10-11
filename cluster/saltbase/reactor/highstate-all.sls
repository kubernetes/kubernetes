# This runs highstate on the ALL nodes.
#
# Used from the vSphere provider. The IP addresses of the minons are passed to
# the apiserver as arguments and every minion has static routes to every other
# minion. This means every node should be refreshed when a node is added.
#
highstate_run:
  cmd.state.highstate:
    - tgt: '*'
