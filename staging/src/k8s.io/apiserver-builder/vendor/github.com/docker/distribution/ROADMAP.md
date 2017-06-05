# Roadmap

The Distribution Project consists of several components, some of which are
still being defined. This document defines the high-level goals of the
project, identifies the current components, and defines the release-
relationship to the Docker Platform.

* [Distribution Goals](#distribution-goals)
* [Distribution Components](#distribution-components)
* [Project Planning](#project-planning): release-relationship to the Docker Platform.

This road map is a living document, providing an overview of the goals and
considerations made in respect of the future of the project.

## Distribution Goals

- Replace the existing [docker registry](github.com/docker/docker-registry)
  implementation as the primary implementation.
- Replace the existing push and pull code in the docker engine with the
  distribution package.
- Define a strong data model for distributing docker images
- Provide a flexible distribution tool kit for use in the docker platform
- Unlock new distribution models

## Distribution Components

Components of the Distribution Project are managed via github [milestones](https://github.com/docker/distribution/milestones). Upcoming
features and bugfixes for a component will be added to the relevant milestone. If a feature or
bugfix is not part of a milestone, it is currently unscheduled for
implementation. 

* [Registry](#registry)
* [Distribution Package](#distribution-package)

***

### Registry

The new Docker registry is the main portion of the distribution repository.
Registry 2.0 is the first release of the next-generation registry. This was
primarily focused on implementing the [new registry
API](https://github.com/docker/distribution/blob/master/docs/spec/api.md),
with a focus on security and performance. 

Following from the Distribution project goals above, we have a set of goals
for registry v2 that we would like to follow in the design. New features
should be compared against these goals.

#### Data Storage and Distribution First

The registry's first goal is to provide a reliable, consistent storage
location for Docker images. The registry should only provide the minimal
amount of indexing required to fetch image data and no more.

This means we should be selective in new features and API additions, including
those that may require expensive, ever growing indexes. Requests should be
servable in "constant time".

#### Content Addressability

All data objects used in the registry API should be content addressable.
Content identifiers should be secure and verifiable. This provides a secure,
reliable base from which to build more advanced content distribution systems.

#### Content Agnostic

In the past, changes to the image format would require large changes in Docker
and the Registry. By decoupling the distribution and image format, we can
allow the formats to progress without having to coordinate between the two.
This means that we should be focused on decoupling Docker from the registry
just as much as decoupling the registry from Docker. Such an approach will
allow us to unlock new distribution models that haven't been possible before.

We can take this further by saying that the new registry should be content
agnostic. The registry provides a model of names, tags, manifests and content
addresses and that model can be used to work with content.

#### Simplicity

The new registry should be closer to a microservice component than its
predecessor. This means it should have a narrower API and a low number of
service dependencies. It should be easy to deploy.

This means that other solutions should be explored before changing the API or
adding extra dependencies. If functionality is required, can it be added as an
extension or companion service.

#### Extensibility

The registry should provide extension points to add functionality. By keeping
the scope narrow, but providing the ability to add functionality.

Features like search, indexing, synchronization and registry explorers fall
into this category. No such feature should be added unless we've found it
impossible to do through an extension.

#### Active Feature Discussions

The following are feature discussions that are currently active.

If you don't see your favorite, unimplemented feature, feel free to contact us
via IRC or the mailing list and we can talk about adding it. The goal here is
to make sure that new features go through a rigid design process before
landing in the registry.

##### Proxying to other Registries

A _pull-through caching_ mode exists for the registry, but is restricted from 
within the docker client to only mirror the official Docker Hub.  This functionality
can be expanded when image provenance has been specified and implemented in the 
distribution project.

##### Metadata storage

Metadata for the registry is currently stored with the manifest and layer data on
the storage backend.  While this is a big win for simplicity and reliably maintaining
state, it comes with the cost of consistency and high latency.  The mutable registry
metadata operations should be abstracted behind an API which will allow ACID compliant
storage systems to handle metadata.

##### Peer to Peer transfer

Discussion has started here: https://docs.google.com/document/d/1rYDpSpJiQWmCQy8Cuiaa3NH-Co33oK_SC9HeXYo87QA/edit

##### Indexing, Search and Discovery

The original registry provided some implementation of search for use with
private registries. Support has been elided from V2 since we'd like to both
decouple search functionality from the registry. The makes the registry
simpler to deploy, especially in use cases where search is not needed, and
let's us decouple the image format from the registry.

There are explorations into using the catalog API and notification system to
build external indexes. The current line of thought is that we will define a
common search API to index and query docker images. Such a system could be run
as a companion to a registry or set of registries to power discovery.

The main issue with search and discovery is that there are so many ways to
accomplish it. There are two aspects to this project. The first is deciding on
how it will be done, including an API definition that can work with changing
data formats. The second is the process of integrating with `docker search`.
We expect that someone attempts to address the problem with the existing tools
and propose it as a standard search API or uses it to inform a standardization
process. Once this has been explored, we integrate with the docker client.

Please see the following for more detail:

- https://github.com/docker/distribution/issues/206

##### Deletes

> __NOTE:__ Deletes are a much asked for feature. Before requesting this
feature or participating in discussion, we ask that you read this section in
full and understand the problems behind deletes.

While, at first glance, implementing deleting seems simple, there are a number
mitigating factors that make many solutions not ideal or even pathological in
the context of a registry. The following paragraph discuss the background and
approaches that could be applied to a arrive at a solution.

The goal of deletes in any system is to remove unused or unneeded data. Only
data requested for deletion should be removed and no other data. Removing
unintended data is worse than _not_ removing data that was requested for
removal but ideally, both are supported. Generally, according to this rule, we
err on holding data longer than needed, ensuring that it is only removed when
we can be certain that it can be removed. With the current behavior, we opt to
hold onto the data forever, ensuring that data cannot be incorrectly removed.

To understand the problems with implementing deletes, one must understand the
data model. All registry data is stored in a filesystem layout, implemented on
a "storage driver", effectively a _virtual file system_ (VFS). The storage
system must assume that this VFS layer will be eventually consistent and has
poor read- after-write consistency, since this is the lower common denominator
among the storage drivers. This is mitigated by writing values in reverse-
dependent order, but makes wider transactional operations unsafe.

Layered on the VFS model is a content-addressable _directed, acyclic graph_
(DAG) made up of blobs. Manifests reference layers. Tags reference manifests.
Since the same data can be referenced by multiple manifests, we only store
data once, even if it is in different repositories. Thus, we have a set of
blobs, referenced by tags and manifests. If we want to delete a blob we need
to be certain that it is no longer referenced by another manifest or tag. When
we delete a manifest, we also can try to delete the referenced blobs. Deciding
whether or not a blob has an active reference is the crux of the problem.

Conceptually, deleting a manifest and its resources is quite simple. Just find
all the manifests, enumerate the referenced blobs and delete the blobs not in
that set. An astute observer will recognize this as a garbage collection
problem. As with garbage collection in programming languages, this is very
simple when one always has a consistent view. When one adds parallelism and an
inconsistent view of data, it becomes very challenging.

A simple example can demonstrate this. Let's say we are deleting a manifest
_A_ in one process. We scan the manifest and decide that all the blobs are
ready for deletion. Concurrently, we have another process accepting a new
manifest _B_ referencing one or more blobs from the manifest _A_. Manifest _B_
is accepted and all the blobs are considered present, so the operation
proceeds. The original process then deletes the referenced blobs, assuming
they were unreferenced. The manifest _B_, which we thought had all of its data
present, can no longer be served by the registry, since the dependent data has
been deleted.

Deleting data from the registry safely requires some way to coordinate this
operation. The following approaches are being considered:

- _Reference Counting_ - Maintain a count of references to each blob. This is
  challenging for a number of reasons: 1. maintaining a consistent consensus
  of reference counts across a set of Registries and 2. Building the initial
  list of reference counts for an existing registry. These challenges can be
  met with a consensus protocol like Paxos or Raft in the first case and a
  necessary but simple scan in the second..
- _Lock the World GC_ - Halt all writes to the data store. Walk the data store
  and find all blob references. Delete all unreferenced blobs. This approach
  is very simple but requires disabling writes for a period of time while the
  service reads all data. This is slow and expensive but very accurate and
  effective.
- _Generational GC_ - Do something similar to above but instead of blocking
  writes, writes are sent to another storage backend while reads are broadcast
  to the new and old backends. GC is then performed on the read-only portion.
  Because writes land in the new backend, the data in the read-only section
  can be safely deleted. The main drawbacks of this approach are complexity
  and coordination.
- _Centralized Oracle_ - Using a centralized, transactional database, we can
  know exactly which data is referenced at any given time. This avoids
  coordination problem by managing this data in a single location. We trade
  off metadata scalability for simplicity and performance. This is a very good
  option for most registry deployments. This would create a bottleneck for
  registry metadata. However, metadata is generally not the main bottleneck
  when serving images.

Please let us know if other solutions exist that we have yet to enumerate.
Note that for any approach, implementation is a massive consideration. For
example, a mark-sweep based solution may seem simple but the amount of work in
coordination offset the extra work it might take to build a _Centralized
Oracle_. We'll accept proposals for any solution but please coordinate with us
before dropping code.

At this time, we have traded off simplicity and ease of deployment for disk
space. Simplicity and ease of deployment tend to reduce developer involvement,
which is currently the most expensive resource in software engineering. Taking
on any solution for deletes will greatly effect these factors, trading off
very cheap disk space for a complex deployment and operational story.

Please see the following issues for more detail:

- https://github.com/docker/distribution/issues/422
- https://github.com/docker/distribution/issues/461
- https://github.com/docker/distribution/issues/462

### Distribution Package 

At its core, the Distribution Project is a set of Go packages that make up
Distribution Components. At this time, most of these packages make up the
Registry implementation. 

The package itself is considered unstable. If you're using it, please take care to vendor the dependent version. 

For feature additions, please see the Registry section. In the future, we may break out a
separate Roadmap for distribution-specific features that apply to more than
just the registry.

***

### Project Planning

An [Open-Source Planning Process](https://github.com/docker/distribution/wiki/Open-Source-Planning-Process) is used to define the Roadmap. [Project Pages](https://github.com/docker/distribution/wiki) define the goals for each Milestone and identify current progress.

