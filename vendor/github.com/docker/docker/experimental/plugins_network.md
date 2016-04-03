# Experimental: Docker network driver plugins

Docker supports network driver plugins via 
[LibNetwork](https://github.com/docker/libnetwork). Network driver plugins are 
implemented as "remote drivers" for LibNetwork, which shares plugin 
infrastructure with Docker. In effect this means that network driver plugins 
are activated in the same way as other plugins, and use the same kind of 
protocol.

## Using network driver plugins

The means of installing and running a network driver plugin will depend on the
particular plugin.

Once running however, network driver plugins are used just like the built-in
network drivers: by being mentioned as a driver in network-oriented Docker
commands. For example,

    docker network create -d weave mynet

Some network driver plugins are listed in [plugins.md](/docs/extend/plugins.md)

The network thus created is owned by the plugin, so subsequent commands
referring to that network will also be run through the plugin.

## Network driver plugin protocol

The network driver protocol, additional to the plugin activation call, is
documented as part of LibNetwork:
[https://github.com/docker/libnetwork/blob/master/docs/remote.md](https://github.com/docker/libnetwork/blob/master/docs/remote.md).

# Related GitHub PRs and issues

Please record your feedback in the following issue, on the usual
Google Groups, or the IRC channel #docker-network.

 - [#14083](https://github.com/docker/docker/issues/14083) Feedback on
   experimental networking features

Other pertinent issues:

 - [#13977](https://github.com/docker/docker/issues/13977) UI for using networks
 - [#14023](https://github.com/docker/docker/pull/14023) --default-network option
 - [#14051](https://github.com/docker/docker/pull/14051) --publish-service option
 - [#13441](https://github.com/docker/docker/pull/13441) (Deprecated) Networks API & UI
