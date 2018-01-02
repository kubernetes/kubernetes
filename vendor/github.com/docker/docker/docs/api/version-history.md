---
title: "Engine API version history"
description: "Documentation of changes that have been made to Engine API."
keywords: "API, Docker, rcli, REST, documentation"
---

<!-- This file is maintained within the docker/docker Github
     repository at https://github.com/docker/docker/. Make all
     pull requests against that repo. If you see this file in
     another repository, consider it read-only there, as it will
     periodically be overwritten by the definitive file. Pull
     requests which include edits to this file in other repositories
     will be rejected.
-->

## v1.31 API changes

[Docker Engine API v1.31](https://docs.docker.com/engine/api/v1.31/) documentation

* `DELETE /secrets/(name)` now returns status code 404 instead of 500 when the secret does not exist.
* `POST /secrets/create` now returns status code 409 instead of 500 when creating an already existing secret.
* `POST /secrets/(name)/update` now returns status code 400 instead of 500 when updating a secret's content which is not the labels.
* `POST /nodes/(name)/update` now returns status code 400 instead of 500 when demoting last node fails.
* `GET /networks/(id or name)` now takes an optional query parameter `scope` that will filter the network based on the scope (`local`, `swarm`, or `global`).
* `POST /session` is a new endpoint that can be used for running interactive long-running protocols between client and
  the daemon. This endpoint is experimental and only available if the daemon is started with experimental features
  enabled.
* `GET /images/(name)/get` now includes an `ImageMetadata` field which contains image metadata that is local to the engine and not part of the image config.
* `POST /services/create` now accepts a `PluginSpec` when `TaskTemplate.Runtime` is set to `plugin`
* `GET /events` now supports config events `create`, `update` and `remove` that are emitted when users create, update or remove a config
* `GET /volumes/` and `GET /volumes/{name}` now return a `CreatedAt` field,
  containing the date/time the volume was created. This field is omitted if the
  creation date/time for the volume is unknown. For volumes with scope "global",
  this field represents the creation date/time of the local _instance_ of the
  volume, which may differ from instances of the same volume on different nodes.
* `GET /system/df` now returns a `CreatedAt` field for `Volumes`. Refer to the
  `/volumes/` endpoint for a description of this field.

## v1.30 API changes

[Docker Engine API v1.30](https://docs.docker.com/engine/api/v1.30/) documentation

* `GET /info` now returns the list of supported logging drivers, including plugins.
* `GET /info` and `GET /swarm` now returns the cluster-wide swarm CA info if the node is in a swarm: the cluster root CA certificate, and the cluster TLS
 leaf certificate issuer's subject and public key. It also displays the desired CA signing certificate, if any was provided as part of the spec.
* `POST /build/` now (when not silent) produces an `Aux` message in the JSON output stream with payload `types.BuildResult` for each image produced. The final such message will reference the image resulting from the build.
* `GET /nodes` and `GET /nodes/{id}` now returns additional information about swarm TLS info if the node is part of a swarm: the trusted root CA, and the
 issuer's subject and public key.
* `GET /distribution/(name)/json` is a new endpoint that returns a JSON output stream with payload `types.DistributionInspect` for an image name. It includes a descriptor with the digest, and supported platforms retrieved from directly contacting the registry.
* `POST /swarm/update` now accepts 3 additional parameters as part of the swarm spec's CA configuration; the desired CA certificate for
 the swarm, the desired CA key for the swarm (if not using an external certificate), and an optional parameter to force swarm to
 generate and rotate to a new CA certificate/key pair.
* `POST /service/create` and `POST /services/(id or name)/update` now take the field `Platforms` as part of the service `Placement`, allowing to specify platforms supported by the service.
* `POST /containers/(name)/wait` now accepts a `condition` query parameter to indicate which state change condition to wait for. Also, response headers are now returned immediately to acknowledge that the server has registered a wait callback for the client.
* `POST /swarm/init` now accepts a `DataPathAddr` property to set the IP-address or network interface to use for data traffic
* `POST /swarm/join` now accepts a `DataPathAddr` property to set the IP-address or network interface to use for data traffic
* `GET /events` now supports service, node and secret events which are emmited when users create, update and remove service, node and secret 
* `GET /events` now supports network remove event which is emmitted when users remove a swarm scoped network
* `GET /events` now supports a filter type `scope` in which supported value could be swarm and local

## v1.29 API changes

[Docker Engine API v1.29](https://docs.docker.com/engine/api/v1.29/) documentation

* `DELETE /networks/(name)` now allows to remove the ingress network, the one used to provide the routing-mesh.
* `POST /networks/create` now supports creating the ingress network, by specifying an `Ingress` boolean field. As of now this is supported only when using the overlay network driver.
* `GET /networks/(name)` now returns an `Ingress` field showing whether the network is the ingress one.
* `GET /networks/` now supports a `scope` filter to filter networks based on the network mode (`swarm`, `global`, or `local`).
* `POST /containers/create`, `POST /service/create` and `POST /services/(id or name)/update` now takes the field `StartPeriod` as a part of the `HealthConfig` allowing for specification of a period during which the container should not be considered unhealthy even if health checks do not pass.
* `GET /services/(id)` now accepts an `insertDefaults` query-parameter to merge default values into the service inspect output.
* `POST /containers/prune`, `POST /images/prune`, `POST /volumes/prune`, and `POST /networks/prune` now support a `label` filter to filter containers, images, volumes, or networks based on the label. The format of the label filter could be `label=<key>`/`label=<key>=<value>` to remove those with the specified labels, or `label!=<key>`/`label!=<key>=<value>` to remove those without the specified labels.
* `POST /services/create` now accepts `Privileges` as part of `ContainerSpec`. Privileges currently include
  `CredentialSpec` and `SELinuxContext`.

## v1.28 API changes

[Docker Engine API v1.28](https://docs.docker.com/engine/api/v1.28/) documentation

* `POST /containers/create` now includes a `Consistency` field to specify the consistency level for each `Mount`, with possible values `default`, `consistent`, `cached`, or `delegated`.
* `GET /containers/create` now takes a `DeviceCgroupRules` field in `HostConfig` allowing to set custom device cgroup rules for the created container.
* Optional query parameter `verbose` for `GET /networks/(id or name)` will now list all services with all the tasks, including the non-local tasks on the given network.
* `GET /containers/(id or name)/attach/ws` now returns WebSocket in binary frame format for API version >= v1.28, and returns WebSocket in text frame format for API version< v1.28, for the purpose of backward-compatibility.
* `GET /networks` is optimised only to return list of all networks and network specific information. List of all containers attached to a specific network is removed from this API and is only available using the network specific `GET /networks/{network-id}.
* `GET /containers/json` now supports `publish` and `expose` filters to filter containers that expose or publish certain ports.
* `POST /services/create` and `POST /services/(id or name)/update` now accept the `ReadOnly` parameter, which mounts the container's root filesystem as read only.
* `POST /build` now accepts `extrahosts` parameter to specify a host to ip mapping to use during the build.
* `POST /services/create` and `POST /services/(id or name)/update` now accept a `rollback` value for `FailureAction`.
* `POST /services/create` and `POST /services/(id or name)/update` now accept an optional `RollbackConfig` object which specifies rollback options.
* `GET /services` now supports a `mode` filter to filter services based on the service mode (either `global` or `replicated`).
* `POST /containers/(name)/update` now supports updating `NanoCPUs` that represents CPU quota in units of 10<sup>-9</sup> CPUs.

## v1.27 API changes

[Docker Engine API v1.27](https://docs.docker.com/engine/api/v1.27/) documentation

* `GET /containers/(id or name)/stats` now includes an `online_cpus` field in both `precpu_stats` and `cpu_stats`. If this field is `nil` then for compatibility with older daemons the length of the corresponding `cpu_usage.percpu_usage` array should be used.

## v1.26 API changes

[Docker Engine API v1.26](https://docs.docker.com/engine/api/v1.26/) documentation

* `POST /plugins/(plugin name)/upgrade` upgrade a plugin.

## v1.25 API changes

[Docker Engine API v1.25](https://docs.docker.com/engine/api/v1.25/) documentation

* The API version is now required in all API calls. Instead of just requesting, for example, the URL `/containers/json`, you must now request `/v1.25/containers/json`.
* `GET /version` now returns `MinAPIVersion`.
* `POST /build` accepts `networkmode` parameter to specify network used during build.
* `GET /images/(name)/json` now returns `OsVersion` if populated
* `GET /info` now returns `Isolation`.
* `POST /containers/create` now takes `AutoRemove` in HostConfig, to enable auto-removal of the container on daemon side when the container's process exits.
* `GET /containers/json` and `GET /containers/(id or name)/json` now return `"removing"` as a value for the `State.Status` field if the container is being removed. Previously, "exited" was returned as status.
* `GET /containers/json` now accepts `removing` as a valid value for the `status` filter.
* `GET /containers/json` now supports filtering containers by `health` status.
* `DELETE /volumes/(name)` now accepts a `force` query parameter to force removal of volumes that were already removed out of band by the volume driver plugin.
* `POST /containers/create/` and `POST /containers/(name)/update` now validates restart policies.
* `POST /containers/create` now validates IPAMConfig in NetworkingConfig, and returns error for invalid IPv4 and IPv6 addresses (`--ip` and `--ip6` in `docker create/run`).
* `POST /containers/create` now takes a `Mounts` field in `HostConfig` which replaces `Binds`, `Volumes`, and `Tmpfs`. *note*: `Binds`, `Volumes`, and `Tmpfs` are still available and can be combined with `Mounts`.
* `POST /build` now performs a preliminary validation of the `Dockerfile` before starting the build, and returns an error if the syntax is incorrect. Note that this change is _unversioned_ and applied to all API versions.
* `POST /build` accepts `cachefrom` parameter to specify images used for build cache.
* `GET /networks/` endpoint now correctly returns a list of *all* networks,
  instead of the default network if a trailing slash is provided, but no `name`
  or `id`.
* `DELETE /containers/(name)` endpoint now returns an error of `removal of container name is already in progress` with status code of 400, when container name is in a state of removal in progress.
* `GET /containers/json` now supports a `is-task` filter to filter
  containers that are tasks (part of a service in swarm mode).
* `POST /containers/create` now takes `StopTimeout` field.
* `POST /services/create` and `POST /services/(id or name)/update` now accept `Monitor` and `MaxFailureRatio` parameters, which control the response to failures during service updates.
* `POST /services/(id or name)/update` now accepts a `ForceUpdate` parameter inside the `TaskTemplate`, which causes the service to be updated even if there are no changes which would ordinarily trigger an update.
* `POST /services/create` and `POST /services/(id or name)/update` now return a `Warnings` array.
* `GET /networks/(name)` now returns field `Created` in response to show network created time.
* `POST /containers/(id or name)/exec` now accepts an `Env` field, which holds a list of environment variables to be set in the context of the command execution.
* `GET /volumes`, `GET /volumes/(name)`, and `POST /volumes/create` now return the `Options` field which holds the driver specific options to use for when creating the volume.
* `GET /exec/(id)/json` now returns `Pid`, which is the system pid for the exec'd process.
* `POST /containers/prune` prunes stopped containers.
* `POST /images/prune` prunes unused images.
* `POST /volumes/prune` prunes unused volumes.
* `POST /networks/prune` prunes unused networks.
* Every API response now includes a `Docker-Experimental` header specifying if experimental features are enabled (value can be `true` or `false`).
* Every API response now includes a `API-Version` header specifying the default API version of the server.
* The `hostConfig` option now accepts the fields `CpuRealtimePeriod` and `CpuRtRuntime` to allocate cpu runtime to rt tasks when `CONFIG_RT_GROUP_SCHED` is enabled in the kernel.
* The `SecurityOptions` field within the `GET /info` response now includes `userns` if user namespaces are enabled in the daemon.
* `GET /nodes` and `GET /node/(id or name)` now return `Addr` as part of a node's `Status`, which is the address that that node connects to the manager from.
* The `HostConfig` field now includes `NanoCPUs` that represents CPU quota in units of 10<sup>-9</sup> CPUs.
* `GET /info` now returns more structured information about security options.
* The `HostConfig` field now includes `CpuCount` that represents the number of CPUs available for execution by the container. Windows daemon only.
* `POST /services/create` and `POST /services/(id or name)/update` now accept the `TTY` parameter, which allocate a pseudo-TTY in container.
* `POST /services/create` and `POST /services/(id or name)/update` now accept the `DNSConfig` parameter, which specifies DNS related configurations in resolver configuration file (resolv.conf) through `Nameservers`, `Search`, and `Options`.
* `GET /networks/(id or name)` now includes IP and name of all peers nodes for swarm mode overlay networks.
* `GET /plugins` list plugins.
* `POST /plugins/pull?name=<plugin name>` pulls a plugin.
* `GET /plugins/(plugin name)` inspect a plugin.
* `POST /plugins/(plugin name)/set` configure a plugin.
* `POST /plugins/(plugin name)/enable` enable a plugin.
* `POST /plugins/(plugin name)/disable` disable a plugin.
* `POST /plugins/(plugin name)/push` push a plugin.
* `POST /plugins/create?name=(plugin name)` create a plugin.
* `DELETE /plugins/(plugin name)` delete a plugin.
* `POST /node/(id or name)/update` now accepts both `id` or `name` to identify the node to update.
* `GET /images/json` now support a `reference` filter.
* `GET /secrets` returns information on the secrets.
* `POST /secrets/create` creates a secret.
* `DELETE /secrets/{id}` removes the secret `id`.
* `GET /secrets/{id}` returns information on the secret `id`.
* `POST /secrets/{id}/update` updates the secret `id`.
* `POST /services/(id or name)/update` now accepts service name or prefix of service id as a parameter.
* `POST /containers/create` added 2 built-in log-opts that work on all logging drivers,
`mode` (`blocking`|`non-blocking`), and `max-buffer-size` (e.g. `2m`) which enables a non-blocking log buffer.

## v1.24 API changes

[Docker Engine API v1.24](v1.24.md) documentation

* `POST /containers/create` now takes `StorageOpt` field.
* `GET /info` now returns `SecurityOptions` field, showing if `apparmor`, `seccomp`, or `selinux` is supported.
* `GET /info` no longer returns the `ExecutionDriver` property. This property was no longer used after integration
  with ContainerD in Docker 1.11.
* `GET /networks` now supports filtering by `label` and `driver`.
* `GET /containers/json` now supports filtering containers by `network` name or id.
* `POST /containers/create` now takes `IOMaximumBandwidth` and `IOMaximumIOps` fields. Windows daemon only.
* `POST /containers/create` now returns an HTTP 400 "bad parameter" message
  if no command is specified (instead of an HTTP 500 "server error")
* `GET /images/search` now takes a `filters` query parameter.
* `GET /events` now supports a `reload` event that is emitted when the daemon configuration is reloaded.
* `GET /events` now supports filtering by daemon name or ID.
* `GET /events` now supports a `detach` event that is emitted on detaching from container process.
* `GET /events` now supports an `exec_detach ` event that is emitted on detaching from exec process.
* `GET /images/json` now supports filters `since` and `before`.
* `POST /containers/(id or name)/start` no longer accepts a `HostConfig`.
* `POST /images/(name)/tag` no longer has a `force` query parameter.
* `GET /images/search` now supports maximum returned search results `limit`.
* `POST /containers/{name:.*}/copy` is now removed and errors out starting from this API version.
* API errors are now returned as JSON instead of plain text.
* `POST /containers/create` and `POST /containers/(id)/start` allow you to configure kernel parameters (sysctls) for use in the container.
* `POST /containers/<container ID>/exec` and `POST /exec/<exec ID>/start`
  no longer expects a "Container" field to be present. This property was not used
  and is no longer sent by the docker client.
* `POST /containers/create/` now validates the hostname (should be a valid RFC 1123 hostname).
* `POST /containers/create/` `HostConfig.PidMode` field now accepts `container:<name|id>`,
  to have the container join the PID namespace of an existing container.

## v1.23 API changes

[Docker Engine API v1.23](v1.23.md) documentation

* `GET /containers/json` returns the state of the container, one of `created`, `restarting`, `running`, `paused`, `exited` or `dead`.
* `GET /containers/json` returns the mount points for the container.
* `GET /networks/(name)` now returns an `Internal` field showing whether the network is internal or not.
* `GET /networks/(name)` now returns an `EnableIPv6` field showing whether the network has ipv6 enabled or not.
* `POST /containers/(name)/update` now supports updating container's restart policy.
* `POST /networks/create` now supports enabling ipv6 on the network by setting the `EnableIPv6` field (doing this with a label will no longer work).
* `GET /info` now returns `CgroupDriver` field showing what cgroup driver the daemon is using; `cgroupfs` or `systemd`.
* `GET /info` now returns `KernelMemory` field, showing if "kernel memory limit" is supported.
* `POST /containers/create` now takes `PidsLimit` field, if the kernel is >= 4.3 and the pids cgroup is supported.
* `GET /containers/(id or name)/stats` now returns `pids_stats`, if the kernel is >= 4.3 and the pids cgroup is supported.
* `POST /containers/create` now allows you to override usernamespaces remapping and use privileged options for the container.
* `POST /containers/create` now allows specifying `nocopy` for named volumes, which disables automatic copying from the container path to the volume.
* `POST /auth` now returns an `IdentityToken` when supported by a registry.
* `POST /containers/create` with both `Hostname` and `Domainname` fields specified will result in the container's hostname being set to `Hostname`, rather than `Hostname.Domainname`.
* `GET /volumes` now supports more filters, new added filters are `name` and `driver`.
* `GET /containers/(id or name)/logs` now accepts a `details` query parameter to stream the extra attributes that were provided to the containers `LogOpts`, such as environment variables and labels, with the logs.
* `POST /images/load` now returns progress information as a JSON stream, and has a `quiet` query parameter to suppress progress details.

## v1.22 API changes

[Docker Engine API v1.22](v1.22.md) documentation

* `POST /container/(name)/update` updates the resources of a container.
* `GET /containers/json` supports filter `isolation` on Windows.
* `GET /containers/json` now returns the list of networks of containers.
* `GET /info` Now returns `Architecture` and `OSType` fields, providing information
  about the host architecture and operating system type that the daemon runs on.
* `GET /networks/(name)` now returns a `Name` field for each container attached to the network.
* `GET /version` now returns the `BuildTime` field in RFC3339Nano format to make it
  consistent with other date/time values returned by the API.
* `AuthConfig` now supports a `registrytoken` for token based authentication
* `POST /containers/create` now has a 4M minimum value limit for `HostConfig.KernelMemory`
* Pushes initiated with `POST /images/(name)/push` and pulls initiated with `POST /images/create`
  will be cancelled if the HTTP connection making the API request is closed before
  the push or pull completes.
* `POST /containers/create` now allows you to set a read/write rate limit for a
  device (in bytes per second or IO per second).
* `GET /networks` now supports filtering by `name`, `id` and `type`.
* `POST /containers/create` now allows you to set the static IPv4 and/or IPv6 address for the container.
* `POST /networks/(id)/connect` now allows you to set the static IPv4 and/or IPv6 address for the container.
* `GET /info` now includes the number of containers running, stopped, and paused.
* `POST /networks/create` now supports restricting external access to the network by setting the `Internal` field.
* `POST /networks/(id)/disconnect` now includes a `Force` option to forcefully disconnect a container from network
* `GET /containers/(id)/json` now returns the `NetworkID` of containers.
* `POST /networks/create` Now supports an options field in the IPAM config that provides options
  for custom IPAM plugins.
* `GET /networks/{network-id}` Now returns IPAM config options for custom IPAM plugins if any
  are available.
* `GET /networks/<network-id>` now returns subnets info for user-defined networks.
* `GET /info` can now return a `SystemStatus` field useful for returning additional information about applications
  that are built on top of engine.

## v1.21 API changes

[Docker Engine API v1.21](v1.21.md) documentation

* `GET /volumes` lists volumes from all volume drivers.
* `POST /volumes/create` to create a volume.
* `GET /volumes/(name)` get low-level information about a volume.
* `DELETE /volumes/(name)` remove a volume with the specified name.
* `VolumeDriver` was moved from `config` to `HostConfig` to make the configuration portable.
* `GET /images/(name)/json` now returns information about an image's `RepoTags` and `RepoDigests`.
* The `config` option now accepts the field `StopSignal`, which specifies the signal to use to kill a container.
* `GET /containers/(id)/stats` will return networking information respectively for each interface.
* The `HostConfig` option now includes the `DnsOptions` field to configure the container's DNS options.
* `POST /build` now optionally takes a serialized map of build-time variables.
* `GET /events` now includes a `timenano` field, in addition to the existing `time` field.
* `GET /events` now supports filtering by image and container labels.
* `GET /info` now lists engine version information and return the information of `CPUShares` and `Cpuset`.
* `GET /containers/json` will return `ImageID` of the image used by container.
* `POST /exec/(name)/start` will now return an HTTP 409 when the container is either stopped or paused.
* `POST /containers/create` now takes `KernelMemory` in HostConfig to specify kernel memory limit.
* `GET /containers/(name)/json` now accepts a `size` parameter. Setting this parameter to '1' returns container size information in the `SizeRw` and `SizeRootFs` fields.
* `GET /containers/(name)/json` now returns a `NetworkSettings.Networks` field,
  detailing network settings per network. This field deprecates the
  `NetworkSettings.Gateway`, `NetworkSettings.IPAddress`,
  `NetworkSettings.IPPrefixLen`, and `NetworkSettings.MacAddress` fields, which
  are still returned for backward-compatibility, but will be removed in a future version.
* `GET /exec/(id)/json` now returns a `NetworkSettings.Networks` field,
  detailing networksettings per network. This field deprecates the
  `NetworkSettings.Gateway`, `NetworkSettings.IPAddress`,
  `NetworkSettings.IPPrefixLen`, and `NetworkSettings.MacAddress` fields, which
  are still returned for backward-compatibility, but will be removed in a future version.
* The `HostConfig` option now includes the `OomScoreAdj` field for adjusting the
  badness heuristic. This heuristic selects which processes the OOM killer kills
  under out-of-memory conditions.

## v1.20 API changes

[Docker Engine API v1.20](v1.20.md) documentation

* `GET /containers/(id)/archive` get an archive of filesystem content from a container.
* `PUT /containers/(id)/archive` upload an archive of content to be extracted to
an existing directory inside a container's filesystem.
* `POST /containers/(id)/copy` is deprecated in favor of the above `archive`
endpoint which can be used to download files and directories from a container.
* The `hostConfig` option now accepts the field `GroupAdd`, which specifies a
list of additional groups that the container process will run as.

## v1.19 API changes

[Docker Engine API v1.19](v1.19.md) documentation

* When the daemon detects a version mismatch with the client, usually when
the client is newer than the daemon, an HTTP 400 is now returned instead
of a 404.
* `GET /containers/(id)/stats` now accepts `stream` bool to get only one set of stats and disconnect.
* `GET /containers/(id)/logs` now accepts a `since` timestamp parameter.
* `GET /info` The fields `Debug`, `IPv4Forwarding`, `MemoryLimit`, and
`SwapLimit` are now returned as boolean instead of as an int. In addition, the
end point now returns the new boolean fields `CpuCfsPeriod`, `CpuCfsQuota`, and
`OomKillDisable`.
* The `hostConfig` option now accepts the fields `CpuPeriod` and `CpuQuota`
* `POST /build` accepts `cpuperiod` and `cpuquota` options

## v1.18 API changes

[Docker Engine API v1.18](v1.18.md) documentation

* `GET /version` now returns `Os`, `Arch` and `KernelVersion`.
* `POST /containers/create` and `POST /containers/(id)/start`allow you to  set ulimit settings for use in the container.
* `GET /info` now returns `SystemTime`, `HttpProxy`,`HttpsProxy` and `NoProxy`.
* `GET /images/json` added a `RepoDigests` field to include image digest information.
* `POST /build` can now set resource constraints for all containers created for the build.
* `CgroupParent` can be passed in the host config to setup container cgroups under a specific cgroup.
* `POST /build` closing the HTTP request cancels the build
* `POST /containers/(id)/exec` includes `Warnings` field to response.
