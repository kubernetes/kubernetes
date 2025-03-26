# Plugin Registration Service

This folder contains a utility, pluginwatcher, for Kubelet to register
different types of node-level plugins such as device plugins or CSI plugins.
It discovers plugins by monitoring inotify events under the directory returned by
kubelet.getPluginsDir(). We will refer to this directory as PluginsDir.

Plugins are expected to implement the gRPC registration service specified in
pkg/kubelet/apis/pluginregistration/v*/api.proto.

## Plugin Discovery

The pluginwatcher service will discover plugins in the PluginDir when they
place a socket in that directory or, at Kubelet start if the socket is already
there.

This socket filename should not start with a '.' as it will be ignored.

To avoid conflicts between different plugins, the recommendation is to use
`<plugin name>[-<some optional string>].sock` as filename. `<plugin name>`
should end with a DNS domain that is unique for the plugin. Each time a plugin
starts, it has to delete old sockets if they exist and listen anew under the
same filename.

## Monitoring Plugin Connection

**Warning**: Monitoring the plugin connection is only supported
for DRA at the moment.

The Kubelet monitors the gRPC connection to a plugin's **service socket** using
a [gRPC stats handler](https://github.com/grpc/grpc-go/blob/master/examples/features/stats_monitoring/README.md).

This enables the Kubelet to:
- Detect when the plugin process has crashed, exited, or restarted
- Trigger cleanup of the plugin’s resources on connection drop
- Cancel pending cleanup if the connection is restored


The **registration socket** is used by the plugin manager. It registers the
plugin when registration socket is created by the plugin and the GetInfo gRPC
call succeeds. It  deregisters the plugin when the socket is removed.
The plugin should be ready to handle gRPC requests over the **service socket**
that it returned in response to the GetInfo call because the kubelet might try
to use the service immediately. Monitoring this service socket is therefore
more accurate for detecting the real availability of the plugin.

### How It Works

Internally, the plugin client configures a gRPC stats handler to observe
`ConnBegin` and `ConnEnd` events on the service socket connection. The
connection is established over a Unix Domain Socket, which provides reliable
semantics - if the connection drops, it definitively indicates that the plugin
closed its end (e.g., due to crash or shutdown).

1. During plugin registration, the Kubelet connects to plugin's
   **service socket** and attaches the stats handler.
2. A long-lived gRPC connection is established and actively monitored.
3. If the plugin process exits and the connection drops, the stats
   handler observes a `ConnEnd` event.
4. This triggers a check whether cleanup is necessary: as long as at
   least one gRPC connection is connected, the plugin is usable and
   no cleanup is required.
5. A gRPC reconnect is initiated immediately after connection loss.
6. When the plugin resumes serving on the same service socket, the connection
   is re-established and a `ConnBegin` event is observed.
7. This cancels any in-progress resource cleanup and restores communication.

### Key Properties

- The plugin is **not** deregistered when the connection drops.
- This model supports multi-container plugin deployments (e.g., CSI-style
  sidecar setups) where the service container may restart independently of
  the registrar container.
- Cleanup is only executed if the connection is not restored before a
  grace period expires.

## Seamless Upgrade

To avoid downtime of a plugin on a node, it would be nice to support running an
old plugin in parallel to the new plugin. When deploying with a DaemonSet,
setting `maxSurge` to a value larger than zero enables such a seamless upgrade.

**Warning**: Such a seamless upgrade is only supported for DRA at the moment.

### In a plugin

*Note*: For DRA, the
[k8s.io/dynamic-resource-allocation](https://pkg.go.dev/k8s.io/dynamic-resource-allocation/kubeletplugin)
helper package offers the `RollingUpdate` option which implements the socket
handling as described in this section.

To support seamless upgrades, each plugin instance must use a unique
socket filename. Otherwise the following could happen:
- The old instance is registered with `plugin.example.com-reg.sock`.
- The new instance starts, unlinks that file, and starts listening on it again.
- In parallel, the kubelet notices the removal and unregisters the plugin
  before probing the new instance, thus breaking the seamless upgrade.

Even if the timing is more favorable and unregistration is avoided, using the
same socket is problematic: if the new instance fails, the kubelet cannot fall
back to the old instance because that old instance is not listening to the
socket that is available under `plugin.example.com-reg.sock`.

This can be achieved in a DaemonSet by passing the UID of the pod into the pod
through the downward API. New instances may try to clean up stale sockets of
older instances, but have to be absolutely sure that those sockets really
aren't in use anymore. Each instance should catch termination signals and clean
up after itself. Then sockets only leak during abnormal events (power loss,
killing with SIGKILL).

Last but not least, both plugin instances must be usable in parallel. It is not
predictable which instance the kubelet will use for which request.

### In the kubelet

For such a seamless upgrade with different sockets per plugin to work reliably,
the handler for the plugin type must track all registered instances. Then if
one of them fails and gets unregistered, it can fall back to some
other. Picking the most recently registered instance is a good heuristic. This
isn't perfect because after a kubelet restart, plugin instances get registered
in a random order. Restarting the kubelet in the middle of an upgrade should be
rare.

At the moment, the following handlers do not support such seamless upgrades:

- The device plugin handler suffers from temporarily removing the extended
  resources during an upgrade. A proposed fix is pending in
  https://github.com/kubernetes/kubernetes/pull/127821.

- The CSI handler [tries to determine which instance is newer](https://github.com/kubernetes/kubernetes/blob/7140b4910c6c1179c9778a7f3bb8037356febd58/pkg/volume/csi/csi_plugin.go#L115-L125) based on the supported version(s) and
  only remembers that one. If that newest instance fails, there is no fallback.

  In practice, most CSI drivers probably all pass [the hard-coded "1.0.0"](https://github.com/kubernetes-csi/node-driver-registrar/blob/27700e2962cd35b9f2336a156146181e5c75399e/cmd/csi-node-driver-registrar/main.go#L72)
  from the csi-node-registrar as supported version, so this version
  selection mechanism isn't used at all.

This supports it:

- DRA

### Deployment

Deploying a plugin with support for seamless upgrades and per-instance socket
filenames is *not* compatible with a kubelet version that does not have support
for seamless upgrades yet. It breaks like this:

- New instance starts, gets registered and replaces the old one.
- Old instance stops, removing its socket.
- The kubelet notices that, unregisters the plugin.
- The plugin handler removes *the new* instance because it ignores the socket path -> no instance left.

Plugin authors either have to assume that the cluster has a recent enough
kubelet or rely on labeling nodes with support. Then the plugin can use one
simple DaemonSet for nodes without support and another, more complex one where
`maxSurge` is increased to enable seamless upgrades on nodes which support it.
No such label is specified at the moment.

## gRPC Service Lifecycle

For any discovered plugin, kubelet will issue a Registration.GetInfo gRPC call
to get plugin type, name, endpoint and supported service API versions.

If any of the following steps in registration fails, on retry registration will
start from scratch:
- Registration.GetInfo is called against socket.
- Validate is called against internal plugin type handler.
- Register is called against internal plugin type handler.
- NotifyRegistrationStatus is called against socket to indicate registration result.

During plugin initialization phase, Kubelet will issue Plugin specific calls
(e.g: DevicePlugin::GetDevicePluginOptions).

Once Kubelet determines that it is ready to use your plugin it will issue a
Registration.NotifyRegistrationStatus gRPC call.

If the plugin removes its socket from the PluginDir this will be interpreted
as a plugin Deregistration. If any of the following steps in deregistration fails,
on retry deregistration will start from scratch:
- Registration.GetInfo is called against socket.
- DeRegisterPlugin is called against internal plugin type handler.


## gRPC Service Overview

Here are the general rules that Kubelet plugin developers should follow:
- Run plugin as 'root' user. Currently creating socket under PluginsDir, a root owned
  directory, requires plugin process to be running as 'root'.

- The plugin name sent during Registration.GetInfo grpc should be unique
  for the given plugin type (CSIPlugin or DevicePlugin).

- The socket path needs to be unique within one directory, in normal case,
  each plugin type has its own sub directory, but the design does support socket file
  under any sub directory of PluginSockDir.

- A plugin should clean up its own socket upon exiting or when a new instance
  comes up. A plugin should NOT remove any sockets belonging to other plugins.

- A plugin should make sure it has service ready for any supported service API
  version listed in the PluginInfo.

- For an example plugin implementation, take a look at example_plugin.go
  included in this directory.


# Kubelet Interface

For any kubelet components using the pluginwatcher module, you will need to
implement the PluginHandler interface defined in the types.go file.

The interface is documented and the implementations are registered with the
pluginwatcher module in kubelet.go by calling AddHandler(pluginType, handler).


The lifecycle follows a simple state machine:

               Validate -> Register -> DeRegister
                  ^          +
                  |          |
                  +--------- +

The pluginwatcher calls the functions with the received plugin name, supported
service API versions and the endpoint to call the plugin on.

The Kubelet component that receives this callback can acknowledge or reject
the plugin according to its own logic, and use the socket path to establish
its service communication with any API version supported by the plugin.
