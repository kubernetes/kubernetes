/*
Package engineapi provides libraries to implement client and server components compatible with the Docker engine.

The client package in github.com/docker/engine-api/client implements all necessary requests to implement the official Docker engine cli.

Create a new client, then use it to send and receive messages to the Docker engine API:

	defaultHeaders := map[string]string{"User-Agent": "engine-api-cli-1.0"}
	cli, err := client.NewClient("unix:///var/run/docker.sock", "v1.22", nil, defaultHeaders)

Other programs, like Docker Machine, can set the default Docker engine environment for you. There is a shortcut to use its variables to configure the client:

	cli, err := client.NewEnvClient()

All request arguments are defined as typed structures in the types package. For instance, this is how to get all containers running in the host:

	options := types.ContainerListOptions{All: true}
	containers, err := cli.ContainerList(options)

*/
package engineapi
