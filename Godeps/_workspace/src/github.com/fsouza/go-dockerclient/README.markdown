#go-dockerclient

[![Build Status](https://drone.io/github.com/fsouza/go-dockerclient/status.png)](https://drone.io/github.com/fsouza/go-dockerclient/latest)
[![Build Status](https://travis-ci.org/fsouza/go-dockerclient.png)](https://travis-ci.org/fsouza/go-dockerclient)

[![GoDoc](https://godoc.org/github.com/fsouza/go-dockerclient?status.png)](https://godoc.org/github.com/fsouza/go-dockerclient)

This package presents a client for the Docker remote API. It also provides
support for the extensions in the [Swarm API](https://docs.docker.com/swarm/API/).

For more details, check the [remote API documentation](http://docs.docker.com/en/latest/reference/api/docker_remote_api/).

## Example

```go
package main

import (
	"fmt"

	"github.com/fsouza/go-dockerclient"
)

func main() {
	endpoint := "unix:///var/run/docker.sock"
	client, _ := docker.NewClient(endpoint)
	imgs, _ := client.ListImages(docker.ListImagesOptions{All: false})
	for _, img := range imgs {
		fmt.Println("ID: ", img.ID)
		fmt.Println("RepoTags: ", img.RepoTags)
		fmt.Println("Created: ", img.Created)
		fmt.Println("Size: ", img.Size)
		fmt.Println("VirtualSize: ", img.VirtualSize)
		fmt.Println("ParentId: ", img.ParentID)
	}
}
```

## Using with Boot2Docker

Boot2Docker runs Docker with TLS enabled. In order to instantiate the client you should use NewTLSClient, passing the endpoint and path for key and certificates as parameters.

For more details about TLS support in Boot2Docker, please refer to [TLS support](https://github.com/boot2docker/boot2docker#tls-support) on Boot2Docker's readme.

```go
package main

import (
	"fmt"

	"github.com/fsouza/go-dockerclient"
)

func main() {
	endpoint := "tcp://[ip]:[port]"
	path := os.Getenv("DOCKER_CERT_PATH")
	ca := fmt.Sprintf("%s/ca.pem", path)
	cert := fmt.Sprintf("%s/cert.pem", path)
	key := fmt.Sprintf("%s/key.pem", path)
	client, _ := docker.NewTLSClient(endpoint, cert, key, ca)
	// use client
}
```

## Developing

You can run the tests with:

    make test
