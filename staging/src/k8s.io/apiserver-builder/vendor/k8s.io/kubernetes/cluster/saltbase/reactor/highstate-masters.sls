# This runs highstate on the master node(s).
#
# Some of the cluster deployment scripts pass the list of minion addresses to
# the apiserver as a command line argument. This list needs to be updated if a
# new minion is started, so run highstate on the master(s) when this happens.
#
highstate_master:
  cmd.state.highstate:
    - tgt: 'roles:kubernetes-master'
    - expr_form: grain
