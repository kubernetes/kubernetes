## Abstract

In the current system, most watch requests sent to apiserver are redirected to
etcd. This means that for every watch request the apiserver opens a watch on
etcd.

The purpose of the proposal is to improve the overall performance of the system
by solving the following problems:

- having too many open watches on etcd
- avoiding deserializing/converting the same objects multiple times in different
watch results

In the future, we would also like to add an indexing mechanism to the watch.
Although Indexer is not part of this proposal, it is supposed to be compatible
with it - in the future Indexer should be incorporated into the proposed new
watch solution in apiserver without requiring any redesign.


## High level design

We are going to solve those problems by allowing many clients to watch the same
storage in the apiserver, without being redirected to etcd.

At the high level, apiserver will have a single watch open to etcd, watching all
the objects (of a given type) without any filtering. The changes delivered from
etcd will then be stored in a cache in apiserver. This cache is in fact a
"rolling history window" that will support clients having some amount of latency
between their list and watch calls. Thus it will have a limited capacity and
whenever a new change comes from etcd when a cache is full, the oldest change
will be remove to make place for the new one.

When a client sends a watch request to apiserver, instead of redirecting it to
etcd, it will cause:

  - registering a handler to receive all new changes coming from etcd
  - iterating though a watch window, starting at the requested resourceVersion
    to the head and sending filtered changes directory to the client, blocking
    the above until this iteration has caught up

This will be done be creating a go-routine per watcher that will be responsible
for performing the above.

The following section describes the proposal in more details, analyzes some
corner cases and divides the whole design in more fine-grained steps.


## Proposal details

We would like the cache to be __per-resource-type__ and __optional__. Thanks to
it we will be able to:
  - have different cache sizes for different resources (e.g. bigger cache
    [= longer history] for pods, which can significantly affect performance)
  - avoid any overhead for objects that are watched very rarely (e.g. events
    are almost not watched at all, but there are a lot of them)
  - filter the cache for each watcher more effectively

If we decide to support watches spanning different resources in the future and
we have an efficient indexing mechanisms, it should be relatively simple to unify
the cache to be common for all the resources.

The rest of this section describes the concrete steps that need to be done
to implement the proposal.

1. Since we want the watch in apiserver to be optional for different resource
types, this needs to be self-contained and hidden behind a well defined API.
This should be a layer very close to etcd - in particular all registries:
"pkg/registry/generic/registry" should be built on top of it.
We will solve it by turning tools.EtcdHelper by extracting its interface
and treating this interface as this API - the whole watch mechanisms in
apiserver will be hidden behind that interface.
Thanks to it we will get an initial implementation for free and we will just
need to reimplement few relevant functions (probably just Watch and List).
Moreover, this will not require any changes in other parts of the code.
This step is about extracting the interface of tools.EtcdHelper.

2. Create a FIFO cache with a given capacity. In its "rolling history window"
we will store two things:

  - the resourceVersion of the object (being an etcdIndex)
  - the object watched from etcd itself (in a deserialized form)

  This should be as simple as having an array an treating it as a cyclic buffer.
  Obviously resourceVersion of objects watched from etcd will be increasing, but
  they are necessary for registering a new watcher that is interested in all the
  changes since a given etcdIndex.

  Additionally, we should support LIST operation, otherwise clients can never
  start watching at now. We may consider passing lists through etcd, however
  this will not work once we have Indexer, so we will need that information
  in memory anyway.
  Thus, we should support LIST operation from the "end of the history" - i.e.
  from the moment just after the newest cached watched event. It should be
  pretty simple to do, because we can incrementally update this list whenever
  the new watch event is watched from etcd.
  We may consider reusing existing structures cache.Store or cache.Indexer
  ("pkg/client/cache") but this is not a hard requirement.

3. Create the new implementation of the API, that will internally have a
single watch open to etcd and will store the data received from etcd in
the FIFO cache - this includes implementing registration of a new watcher
which will start a new go-routine responsible for iterating over the cache
and sending all the objects watcher is interested in (by applying filtering
function) to the watcher.

4. Add a support for processing "error too old" from etcd, which will require:
  - disconnect all the watchers
  - clear the internal cache and relist all objects from etcd
  - start accepting watchers again

5. Enable watch in apiserver for some of the existing resource types - this
should require only changes at the initialization level.

6. The next step will be to incorporate some indexing mechanism, but details
of it are TBD.



### Future optimizations:

1. The implementation of watch in apiserver internally will open a single
watch to etcd, responsible for watching all the changes of objects of a given
resource type. However, this watch can potentially expire at any time and
reconnecting can return "too old resource version". In that case relisting is
necessary. In such case, to avoid LIST requests coming from all watchers at
the same time, we can introduce an additional etcd event type:
[EtcdResync](../../pkg/storage/etcd/etcd_watcher.go#L36)

  Whenever relisting will be done to refresh the internal watch to etcd,
  EtcdResync event will be send to all the watchers. It will contain the
  full list of all the objects the watcher is interested in (appropriately
  filtered) as the parameter of this watch event.
  Thus, we need to create the EtcdResync event, extend watch.Interface and
  its implementations to support it and handle those events appropriately
  in places like
  [Reflector](../../pkg/client/cache/reflector.go)

  However, this might turn out to be unnecessary optimization if apiserver
  will always keep up (which is possible in the new design). We will work
  out all necessary details at that point.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/apiserver-watch.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
