# containerd Namespaces and Multi-Tenancy

containerd offers a fully namespaced API so multiple consumers can all use a single containerd instance without conflicting with one another.
Namespaces allow multi-tenancy within a single daemon. This removes the need for the common pattern of using nested containers to achieve this separation.
Consumers are able to have containers with the same names but with settings and/or configurations that vary drastically.
For example, system or infrastructure level containers can be hidden in one namespace while user level containers are kept in another.
Underlying image content is still shared via content addresses but image names and metadata are separate per namespace.

It is important to note that namespaces, as implemented, is an administrative construct that is not meant to be used as a security feature.
It is trivial for clients to switch namespaces.

## Who specifies the namespace?

The client specifies the namespace via the `context`.
There is a `github.com/containerd/containerd/namespaces` package that allows a user to get and set the namespace on a context.

```go
// set a namespace
ctx := namespaces.WithNamespace(context.Background(), "my-namespace")

// get the namespace
ns, ok := namespaces.Namespace(ctx)
```

Because the client calls containerd's gRPC API to interact with the daemon, all API calls require a context with a namespace set.

## How low level is the implementation?

Namespaces are passed through the containerd API to the underlying plugins providing functionality.
Plugins must be written to take namespaces into account.
Filesystem paths, IDs, and other system level resources must be namespaced for a plugin to work properly.

## How does multi-tenancy work?

Simply create a new `context` and set your application's namespace on the `context`.
Make sure to use a unique namespace for applications that does not conflict with existing namespaces. The namespaces
API, or the `ctr namespaces` client command, can be used to query/list and create new namespaces. Note that namespaces
can have a list of labels associated with the namespace. This can be useful for associating metadata with a particular
namespace.

```go
ctx := context.Background()

var (
	docker = namespaces.WithNamespace(ctx, "docker")
	vmware = namespaces.WithNamespace(ctx, "vmware")
	ecs = namespaces.WithNamespace(ctx, "aws-ecs")
	cri = namespaces.WithNamespace(ctx, "cri")
)
```

## Inspecting Namespaces

If we need to inspect containers, images, or other resources in various namespaces the `ctr` tool allows you to do this.
Simply set the `--namespace,-n` flag on `ctr` to change the namespace. If you do not provide a namespace, `ctr` client commands
will all use the the default namespace, which is simply named "`default`".

```bash
> sudo ctr -n docker tasks
> sudo ctr -n cri tasks
```

You can also use the `CONTAINERD_NAMESPACE` environment variable to specify the default namespace to use for
any of the `ctr` client commands.
