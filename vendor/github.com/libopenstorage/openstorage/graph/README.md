## Graph Driver

This implements the [Graph Driver interface](https://github.com/docker/docker/blob/master/daemon/graphdriver/driver.go).
It allows for different implementations of the interface to be used in conjunction with Docker.  `proxy` is a bypass implementation that simply refers to the `Overlay Graph Driver`.

### Layer0
`layer0` is a graph driver that provides persistent storage for the writable layer of a container.  It can be initialized to use any of the volume drivers to actually provide the persistence.
