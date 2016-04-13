# Experimental: Networking and Services

In this feature:

- `network` and `service` become first class objects in the Docker UI
  - one can now create networks, publish services on that network and attach containers to the services
- Native multi-host networking
  - `network` and `service` objects are globally significant and provides multi-host container connectivity natively
- Inbuilt simple Service Discovery
  - With multi-host networking and top-level `service` object, Docker now provides out of the box simple Service Discovery for containers running in a network
- Batteries included but removable
  - Docker provides inbuilt native multi-host networking by default & can be swapped by any remote driver provided by external plugins.

This is an experimental feature. For information on installing and using experimental features, see [the experimental feature overview](README.md).

## Using Networks

        Usage: docker network [OPTIONS] COMMAND [OPTIONS] [arg...]

        Commands:
            create                   Create a network
            rm                       Remove a network
            ls                       List all networks
            info                     Display information of a network

        Run 'docker network COMMAND --help' for more information on a command.

          --help=false       Print usage

The `docker network` command is used to manage Networks.

To create a network, `docker network create foo`. You can also specify a driver
if you have loaded a networking plugin e.g `docker network create -d <plugin_name> foo`

        $ docker network create foo
        aae601f43744bc1f57c515a16c8c7c4989a2cad577978a32e6910b799a6bccf6
        $ docker network create -d overlay bar
        d9989793e2f5fe400a58ef77f706d03f668219688ee989ea68ea78b990fa2406

`docker network ls` is used to display the currently configured networks

        $ docker network ls
        NETWORK ID          NAME                TYPE
        d367e613ff7f        none                null
        bd61375b6993        host                host
        cc455abccfeb        bridge              bridge
        aae601f43744        foo                 bridge
        d9989793e2f5        bar                 overlay

To get detailed information on a network, you can use the `docker network info`
command.

        $ docker network info foo
        Network Id: aae601f43744bc1f57c515a16c8c7c4989a2cad577978a32e6910b799a6bccf6
        Name: foo
        Type: null

If you no longer have need of a network, you can delete it with `docker network rm`

        $ docker network rm bar
        bar
        $ docker network ls
        NETWORK ID          NAME                TYPE
        aae601f43744        foo                 bridge
        d367e613ff7f        none                null
        bd61375b6993        host                host
        cc455abccfeb        bridge              bridge

## User-Defined default network

Docker daemon supports a configuration flag `--default-network` which takes configuration value of format `DRIVER:NETWORK`, where,
`DRIVER` represents the in-built drivers such as bridge, overlay, container, host and none. or Remote drivers via Network Plugins.
`NETWORK` is the name of the network created using the `docker network create` command
When a container is created and if the network mode (`--net`) is not specified, then this default network will be used to connect
the container. If `--default-network` is not specified, the default network will be the `bridge` driver.
Example : `docker -d --default-network=overlay:multihost`

## Using Services

        Usage: docker service COMMAND [OPTIONS] [arg...]

        Commands:
            publish   Publish a service
            unpublish Remove a service
            attach    Attach a backend (container) to the service
            detach    Detach the backend from the service
            ls        Lists all services
            info      Display information about a service

        Run 'docker service COMMAND --help' for more information on a command.

          --help=false       Print usage

Assuming we want to publish a service from container `a0ebc12d3e48` on network `foo` as `my-service` we would use the following command:

        $ docker service publish my-service.foo
        ec56fd74717d00f968c26675c9a77707e49ae64b8e54832ebf78888eb116e428
        $ docker service attach a0ebc12d3e48 my-service.foo

This would make the container `a0ebc12d3e48` accessible as `my-service` on network `foo`. Any other container in network `foo` can use DNS to resolve the address of `my-service`

This can also be acheived by using the `--publish-service` flag for `docker run`:

        docker run -itd --publish-service db.foo postgres

`db.foo` in this instance means "place the container on network `foo`, and allow other hosts on `foo` to discover it under the name `db`"

We can see the current services using the `docker service ls` command

        $ docker service ls
        SERVICE ID          NAME                NETWORK             PROVIDER
        ec56fd74717d        my-service          foo                 a0ebc12d3e48

To remove the a service:

        $ docker service detach a0ebc12d3e48 my-service.foo
        $ docker service unpublish my-service.foo


## Native Multi-host networking

There is a lot to talk about the native multi-host networking and the `overlay` driver that makes it happen. The technical details are documented under https://github.com/docker/libnetwork/blob/master/docs/overlay.md.
Using the above experimental UI `docker network`, `docker service` and `--publish-service`, the user can exercise the power of multi-host networking.

Since `network` and `service` objects are globally significant, this feature requires distributed states provided by the `libkv` project.
Using `libkv`, the user can plug any of the supported Key-Value store (such as consul, etcd or zookeeper).
User can specify the Key-Value store of choice using the `--kv-store` daemon flag, which takes configuration value of format `PROVIDER:URL`, where
`PROVIDER` is the name of the Key-Value store (such as consul, etcd or zookeeper) and
`URL` is the url to reach the Key-Value store.
Example : `docker -d --kv-store=consul:localhost:8500`


Send us feedback and comments on [#14083](https://github.com/docker/docker/issues/14083)
or on the usual Google Groups (docker-user, docker-dev) and IRC channels.

