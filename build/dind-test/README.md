This is a virtualized cluster in nested docker. The nodes are run as containers,
which in turn run both kubelet and docker. The cluster is a set of these virtual
nodes, which all live in a top-level container.

This is a test platform, and not meant for production systems.
