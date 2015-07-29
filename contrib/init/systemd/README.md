What these give you
------------------------------------

These 'config' files default to launch a single master/node on the same system talking to each
other via 127.0.0.1.

They require that etcd be available at 127.0.0.1:4001.

Daemons may have multiple config files.  An example is that the scheduler will pull in 'config', 'apiserver', and 'scheduler'.  In that order.  Each file may overwrite the values of the previous file.  The 'config' file is sourced by all daemons.  The kube-apiserver config file is sourced by those daemons which must know how to reach the kube-apiserver.  Each daemon has its own config file for configuration specific to that daemon.

Commenting out all values or removing all environment files will launch the daemons with no command line options.

Assumptions of the service files
--------------------------------

1. All binaries live in /usr/bin.
2. There is a user named 'kube' on the system.
   * kube-apiserver, kube-controller-manager, and kube-scheduler are run as kube, not root
3. Configuration is done in via environment files in /etc/kubernetes/

Non kubernetes defaults in the environment files
------------------------------------------------
1. Default to log to stdout/journald instead of directly to disk, see: [KUBE_LOGTOSTDERR](environ/config)
2. Node list of 127.0.0.1 forced instead of relying on cloud provider, see: [KUBELET_ADDRESSES](environ/apiserver)
3. Explicitly set the minion hostname to 127.0.0.1, see: [KUBELET_HOSTNAME](environ/kubelet)
4. There is no default for the IP address range of services.  This uses 10.254.0.0/16 see: [KUBE_SERVICE_ADDRESSES](environ/apiserver)

Notes
-----
It may seem reasonable to use --option=${OPTION} in the .service file instead of only putting the command line option in the environment file.  However this results in the possibility of daemons being called with --option= if the environment file does not define a value.  Whereas including the --option string inside the environment file means that nothing will be passed to the daemon.  So the daemon default will be used for things unset by the environment files.

While some command line options to the daemons use the default when passed an empty option some cause the daemon to fail to launch.  --allow-privileged= (without a value of true/false) will cause the kube-apiserver and kubelet to refuse to launch.

It also may seem reasonable to just use $DAEMON_ARGS and string all of these into one line in the environment file.  While that makes the .service file simple it makes the admin job more difficult to locate and make appropriate changes to the config.  This is a tradeoff between having to update the .service file to add new options or having the config files easy for an admin to work with.  I choose: "easy for admin most of the time".


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/contrib/init/systemd/README.md?pixel)]()
