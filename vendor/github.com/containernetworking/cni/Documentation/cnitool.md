# Overview
The `cnitool` is a utility that can be used to test a CNI plugin
without the need for a container runtime. The `cnitool` takes a
`network name` and a `network namespace` and a command to `ADD` or
`DEL`,.i.e, attach or detach containers from a network. The `cnitool`
relies on the following environment variables to operate properly:
* `NETCONFPATH`: This environment variable needs to be set to a
  directory. It defaults to `/etc/cni/net.d`. The `cnitool` searches
  for CNI configuration files in this directory with the extension
  `*.conf` or `*.json`. It loads all the CNI configuration files in
  this directory and if it finds a CNI configuration with the `network
  name` given to the cnitool it returns the corresponding CNI
  configuration, else it returns `nil`.

* `CNI_PATH`: For a given CNI configuration `cnitool` will search for
  the corresponding CNI plugin in this path.
