# Client Options

The containerd client was built to be easily extended by consumers.
The goal is that the execution flow of the calls remain the same across implementations while `Opts` are written to extend functionality.
To accomplish this we depend on the `Opts` pattern in Go.

## Method Calls

For many functions and methods within the client package you will generally see variadic args as the last parameter.

If we look at the `NewContainer` method on the client we can see that it has a required argument of `id` and then additional `NewContainerOpts`.

There are a few built in options that allow the container to be created with an existing spec, `WithSpec`, and snapshot opts for creating or using an existing snapshot.

```go
func (c *Client) NewContainer(ctx context.Context, id string, opts ...NewContainerOpts) (Container, error) {
}
```

## Extending the Client

As a consumer of the containerd client you need to be able add your domain specific functionality.
There are a few ways of doing this, changing the client code, submitting a PR to the containerd client, or forking the client.
These ways of extending the client should only be considered after every other method has been tried.

The proper and supported way of extending the client is to build a package of `Opts` that define your application specific logic.

As an example, if Docker is integrating containerd support and needs to add concepts such as Volumes, they would create a `docker` package with options.

#### Bad Extension Example

```go
// example code
container, err := client.NewContainer(ctx, id)

// add volumes with their config and bind mounts
container.Labels["volumes"] = VolumeConfig{}
container.Spec.Binds  = append({"/var/lib/docker/volumes..."})
```

#### Good Extension Example

```go
// example code
import "github.com/docker/docker"
import "github.com/docker/libnetwork"

container, err := client.NewContainer(ctx, id,
	docker.WithVolume("volume-name"),
	libnetwork.WithOverlayNetwork("cluster-network"),
)
```

There are a few advantages using this model.

1. Your application code is not scattered in the execution flow of the containerd client.
2. Your code can be unit tested without mocking the containerd client.
3. Contributors can better follow your containerd implementation and understand when and where your application logic is added to standard containerd client calls.

## Example SpecOpt

If we want to make a `SpecOpt` to setup a container to monitor the host system with `htop` it can be easily done without ever touching a line of code in the containerd repository.

```go
package monitor

import (
	"github.com/containerd/containerd"
	specs "github.com/opencontainers/runtime-spec/specs-go"
)

// WithHtop configures a container to monitor the host system via `htop`
func WithHtop(s *specs.Spec) error {
	// make sure we are in the host pid namespace
	if err := containerd.WithHostNamespace(specs.PIDNamespace)(s); err != nil {
		return err
	}
	// make sure we set htop as our arg
	s.Process.Args = []string{"htop"}
	// make sure we have a tty set for htop
	if err := containerd.WithTTY(s); err != nil {
		return err
	}
	return nil
}
```

Adding your new option to spec generation is as easy as importing your new package and adding the option when creating a spec.

```go
import "github.com/crosbymichael/monitor"

container, err := client.NewContainer(ctx, id,
	containerd.WithNewSpec(containerd.WithImageConfig(image), monitor.WithHtop),
)
```

You can see the full code and run the monitor container [here](https://github.com/crosbymichael/monitor).
