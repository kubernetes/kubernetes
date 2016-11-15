<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Client: layering and package structure](#client-layering-and-package-structure)
  - [Desired layers](#desired-layers)
    - [Transport](#transport)
    - [RESTClient/request.go](#restclientrequestgo)
    - [Mux layer](#mux-layer)
    - [High-level: Individual typed](#high-level-individual-typed)
      - [High-level, typed: Discovery](#high-level-typed-discovery)
    - [High-level: Dynamic](#high-level-dynamic)
    - [High-level: Client Sets](#high-level-client-sets)
  - [Package Structure](#package-structure)
  - [Client Guarantees (and testing)](#client-guarantees-and-testing)

<!-- END MUNGE: GENERATED_TOC -->

# Client: layering and package structure

## Desired layers

### Transport

The transport layer is concerned with round-tripping requests to an apiserver
somewhere. It consumes a Config object with options appropriate for this.
(That's most of the current client.Config structure.)

Transport delivers an object that implements http's RoundTripper interface
and/or can be used in place of http.DefaultTransport to route requests.

Transport objects are safe for concurrent use, and are cached and reused by
subsequent layers.

Tentative name: "Transport".

It's expected that the transport config will be general enough that third
parties (e.g., OpenShift) will not need their own implementation, rather they
can change the certs, token, etc., to be appropriate for their own servers,
etc..

Action items:
* Split out of current client package into a new package. (@krousey)

### RESTClient/request.go

RESTClient consumes a Transport and a Codec (and optionally a group/version),
and produces something that implements the interface currently in request.go.
That is, with a RESTClient, you can write chains of calls like:

`c.Get().Path(p).Param("name", "value").Do()`

RESTClient is generically usable by any client for servers exposing REST-like
semantics. It provides helpers that benefit those following api-conventions.md,
but does not mandate them. It provides a higher level http interface that
abstracts transport, wire serialization, retry logic, and error handling.
Kubernetes-like constructs that deviate from standard HTTP should be bypassable.
Every non-trivial call made to a remote restful API from Kubernetes code should
go through a rest client.

The group and version may be empty when constructing a RESTClient. This is valid
for executing discovery commands. The group and version may be overridable with
a chained function call.

Ideally, no semantic behavior is built into RESTClient, and RESTClient will use
the Codec it was constructed with for all semantic operations, including turning
options objects into URL query parameters. Unfortunately, that is not true of
today's RESTClient, which may have some semantic information built in. We will
remove this.

RESTClient should not make assumptions about the format of data produced or
consumed by the Codec. Currently, it is JSON, but we want to support binary
protocols in the future.

The Codec would look something like this:

```go
type Codec interface {
  Encode(runtime.Object) ([]byte, error)
  Decode([]byte]) (runtime.Object, error)

  // Used to version-control query parameters
  EncodeParameters(optionsObject runtime.Object) (url.Values, error)

  // Not included here since the client doesn't need it, but a corresponding
  // DecodeParametersInto method would be available on the server.
}
```

There should be one codec per version. RESTClient is *not* responsible for
converting between versions; if a client wishes, they can supply a Codec that
does that. But RESTClient will make the assumption that it's talking to a single
group/version, and will not contain any conversion logic. (This is a slight
change from the current state.)

As with Transport, it is expected that 3rd party providers following the api
conventions should be able to use RESTClient, and will not need to implement
their own.

Action items:
* Split out of the current client package. (@krousey)
* Possibly, convert to an interface (currently, it's a struct). This will allow
  extending the error-checking monad that's currently in request.go up an
  additional layer.
* Switch from ParamX("x") functions to using types representing the collection
  of parameters and the Codec for query parameter serialization.
* Any other Kubernetes group specific behavior should also be removed from
  RESTClient.

### Mux layer

(See TODO at end; this can probably be merged with the "client set" concept.)

The client muxer layer has a map of group/version to cached RESTClient, and
knows how to construct a new RESTClient in case of a cache miss (using the
discovery client mentioned below). The ClientMux may need to deal with multiple
transports pointing at differing destinations (e.g. OpenShift or other 3rd party
provider API may be at a different location).

When constructing a RESTClient generically, the muxer will just use the Codec
the high-level dynamic client would use. Alternatively, the user should be able
to pass in a Codec-- for the case where the correct types are compiled in.

Tentative name: ClientMux

Action items:
* Move client cache out of kubectl libraries into a more general home.
* TODO: a mux layer may not be necessary, depending on what needs to be cached.
  If transports are cached already, and RESTClients are extremely light-weight,
  there may not need to be much code at all in this layer.

### High-level: Individual typed

Our current high-level client allows you to write things like
`c.Pods("namespace").Create(p)`; we will insert a level for the group.

That is, the system will be:

`clientset.GroupName().NamespaceSpecifier().Action()`

Where:
* `clientset` is a thing that holds multiple individually typed clients (see
  below).
* `GroupName()` returns the generated client that this section is about.
* `NamespaceSpecifier()` may take a namespace parameter or nothing.
* `Action` is one of Create/Get/Update/Delete/Watch, or appropriate actions
  from the type's subresources.
* It is TBD how we'll represent subresources and their actions. This is
  inconsistent in the current clients, so we'll need to define a consistent
  format. Possible choices:
 * Insert a `.Subresource()` before the `.Action()`
 * Flatten subresources, such that they become special Actions on the parent
   resource.

The types returned/consumed by such functions will be e.g. api/v1, NOT the
current version inspecific types. The current internal-versioned client is
inconvenient for users, as it does not protect them from having to recompile
their code with every minor update. (We may continue to generate an
internal-versioned client for our own use for a while, but even for our own
components it probably makes sense to switch to specifically versioned clients.)

We will provide this structure for each version of each group. It is infeasible
to do this manually, so we will generate this. The generator will accept both
swagger and the ordinary go types. The generator should operate on out-of-tree
sources AND out-of-tree destinations, so it will be useful for consuming
out-of-tree APIs and for others to build custom clients into their own
repositories.

Typed clients will be constructable given a ClientMux; the typed constructor will use
the ClientMux to find or construct an appropriate RESTClient. Alternatively, a
typed client should be constructable individually given a config, from which it
will be able to construct the appropriate RESTClient.

Typed clients do not require any version negotiation. The server either supports
the client's group/version, or it does not. However, there are ways around this:
* If you want to use a typed client against a server's API endpoint and the
  server's API version doesn't match the client's API version, you can construct
  the client with a RESTClient using a Codec that does the conversion (this is
  basically what our client does now).
* Alternatively, you could use the dynamic client.

Action items:
* Move current typed clients into new directory structure (described below)
* Finish client generation logic. (@caesarxuchao, @lavalamp)

#### High-level, typed: Discovery

A `DiscoveryClient` is necessary to discover the api groups, versions, and
resources a server supports. It's constructable given a RESTClient. It is
consumed by both the ClientMux and users who want to iterate over groups,
versions, or resources. (Example: namespace controller.)

The DiscoveryClient is *not* required if you already know the group/version of
the resource you want to use: you can simply try the operation without checking
first, which is lower-latency anyway as it avoids an extra round-trip.

Action items:
* Refactor existing functions to present a sane interface, as close to that
  offered by the other typed clients as possible. (@caeserxuchao)
* Use a RESTClient to make the necessary API calls.
* Make sure that no discovery happens unless it is explicitly requested. (Make
  sure SetKubeDefaults doesn't call it, for example.)

### High-level: Dynamic

The dynamic client lets users consume apis which are not compiled into their
binary. It will provide the same interface as the typed client, but will take
and return `runtime.Object`s instead of typed objects. There is only one dynamic
client, so it's not necessary to generate it, although optionally we may do so
depending on whether the typed client generator makes it easy.

A dynamic client is constructable given a config, group, and version. It will
use this to construct a RESTClient with a Codec which encodes/decodes to
'Unstructured' `runtime.Object`s. The group and version may be from a previous
invocation of a DiscoveryClient, or they may be known by other means.

For now, the dynamic client will assume that a JSON encoding is allowed. In the
future, if we have binary-only APIs (unlikely?), we can add that to the
discovery information and construct an appropriate dynamic Codec.

Action items:
* A rudimentary version of this exists in kubectl's builder. It needs to be
  moved to a more general place.
* Produce a useful 'Unstructured' runtime.Object, which allows for easy
  Object/ListMeta introspection.

### High-level: Client Sets

Because there will be multiple groups with multiple versions, we will provide an
aggregation layer that combines multiple typed clients in a single object.

We do this to:
* Deliver a concrete thing for users to consume, construct, and pass around. We
  don't want people making 10 typed clients and making a random system to keep
  track of them.
* Constrain the testing matrix. Users can generate a client set at their whim
  against their cluster, but we need to make guarantees that the clients we
  shipped with v1.X.0 will work with v1.X+1.0, and vice versa. That's not
  practical unless we "bless" a particular version of each API group and ship an
  official client set with earch release. (If the server supports 15 groups with
  2 versions each, that's 2^15 different possible client sets. We don't want to
  test all of them.)

A client set is generated into its own package. The generator will take the list
of group/versions to be included. Only one version from each group will be in
the client set.

A client set is constructable at runtime from either a ClientMux or a transport
config (for easy one-stop-shopping).

An example:

```go
import (
  api_v1 "k8s.io/kubernetes/pkg/client/typed/generated/v1"
  ext_v1beta1 "k8s.io/kubernetes/pkg/client/typed/generated/extensions/v1beta1"
  net_v1beta1 "k8s.io/kubernetes/pkg/client/typed/generated/net/v1beta1"
  "k8s.io/kubernetes/pkg/client/typed/dynamic"
)

type Client interface {
  API() api_v1.Client
  Extensions() ext_v1beta1.Client
  Net() net_v1beta1.Client
  // ... other typed clients here.

  // Included in every set
  Discovery() discovery.Client
  GroupVersion(group, version string) dynamic.Client
}
```

Note that a particular version is chosen for each group. It is a general rule
for our API structure that no client need care about more than one version of
each group at a time.

This is the primary deliverable that people would consume. It is also generated.

Action items:
* This needs to be built. It will replace the ClientInterface that everyone
  passes around right now.

## Package Structure

```
pkg/client/
----------/transport/     # transport & associated config
----------/restclient/
----------/clientmux/
----------/typed/
----------------/discovery/
----------------/generated/
--------------------------/<group>/
----------------------------------/<version>/
--------------------------------------------/<resource>.go
----------------/dynamic/
----------/clientsets/
---------------------/release-1.1/
---------------------/release-1.2/
---------------------/the-test-set-you-just-generated/
```

`/clientsets/` will retain their contents until they reach their expire date.
e.g., when we release v1.N, we'll remove clientset v1.(N-3). Clients from old
releases live on and continue to work (i.e., are tested) without any interface
changes for multiple releases, to give users time to transition.

## Client Guarantees (and testing)

Once we release a clientset, we will not make interface changes to it. Users of
that client will not have to change their code until they are deliberately
upgrading their import. We probably will want to generate some sort of stub test
with a clientset, to ensure that we don't change the interface.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/client-package-structure.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
