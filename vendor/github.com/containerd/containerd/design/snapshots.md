# Snapshots

Docker containers, from the beginning, have long been built on a snapshotting
methodology known as _layers_. _Layers_ provide the ability to fork a
filesystem, make changes then save the changeset back to a new layer.

Historically, these have been tightly integrated into the Docker daemon as a
component called the `graphdriver`. The `graphdriver` allows one to run the
docker daemon on several different operating systems while still maintaining
roughly similar snapshot semantics for committing and distributing changes to
images.

The `graphdriver` is deeply integrated with the import and export of images,
including managing layer relationships and container runtime filesystems. The
behavior of the `graphdriver` informs the transport of image formats.

In this document, we propose a more flexible model for managing layers. It
focuses on providing an API for the base snapshotting functionality without
coupling so tightly to the structure of images and their identification. The
minimal API simplifies behavior without sacrificing power. This makes the
surface area for driver implementations smaller, ensuring that behavior is more
consistent between implementations.

These differ from the concept of the graphdriver in that the _Snapshotter_
has no knowledge of images or containers. Users simply prepare and commit
directories. We also avoid the integration between graph drivers and the tar
format used to represent the changesets.

The best aspect is that we can get to this model by refactoring the existing
graphdrivers, minimizing the need for new code and sprawling tests.

## Scope

In the past, the `graphdriver` component has provided quite a lot of
functionality in Docker. This includes serialization, hashing, unpacking,
packing, mounting.

The _Snapshotter_ will only provide mount-oriented snapshot
access with minimal metadata. Serialization, hashing, unpacking, packing and
mounting are not included in this design, opting for common implementations
between graphdrivers, rather than specialized ones. This is less of a problem
for performance since direct access to changesets is provided in the
interface.

## Architecture

The _Snapshotter_ provides an API for allocating, snapshotting and mounting
abstract, layer-based filesystems. The model works by building up sets of
directories with parent-child relationships, known as _Snapshots_.

A _Snapshot_ represents a filesystem state.  Every snapshot has a parent,
where the empty parent is represented by the empty string.  A diff can be taken
between a parent and its snapshot to create a classic layer.

Snapshots are best understood by their lifecycle.  _Active_ snapshots are always
created with `Prepare` or `View` from a _Committed_ snapshot (including the
empty snapshot).  _Committed_ snapshots are always created with
`Commit` from an _Active_ snapshot.  Active snapshots never become committed
snapshots and vice versa. All snapshots may be removed.

After mounting an _Active_ snapshot, changes can be made to the snapshot.  The
act of committing creates a _Committed_ snapshot.  The committed snapshot will
inherit the parent of the active snapshot.  The committed snapshot can then be
used as a parent.  Active snapshots can never be used as a parent.

The following diagram demonstrates the relationships of snapshots:

![snapshot model diagram, showing active snapshots on the left and
committed snapshots on the right](snapshot_model.png)

In this diagram, you can see that the active snapshot _a_ is created by calling
`Prepare` with the committed snapshot _P<sub>0</sub>_.  After modification, _a_
becomes _a'_ and a committed snapshot _P<sub>1</sub>_ is created by calling
`Commit`.  _a'_ can be further modified as _a''_ and a second committed snapshot
can be created as _P<sub>2</sub>_ by calling `Commit` again.  Note here that
_P<sub>2</sub>_'s parent is _P<sub>0</sub>_ and not _P<sub>1</sub>_.

### Operations

The manifestation of _snapshots_ is facilitated by the `Mount` object and
user-defined directories used for opaque data storage. When creating a new
active snapshot, the caller provides an identifier called the _key_. This
operation returns a list of mounts that, if mounted, will have the fully
prepared snapshot at the mounted path. We call this the _prepare_ operation.

Once a snapshot is _prepared_ and mounted, the caller may write new data to the
snapshot. Depending on the application, a user may want to capture these changes
or not.

For a read-only view of a snapshot, the _view_ operation can be used. Like
_prepare_, _view_ will return a list of mounts that, if mounted, will have the
fully prepared snapshot at the mounted path.

If the user wants to keep the changes, the _commit_ operation is employed. The
_commit_ operation takes the _key_ identifier, which represents an active
snapshot, and a _name_ identifier. A successful result will create a _committed_
snapshot that can be used as the parent of new _active_ snapshots when
referenced by the _name_.

If the user wants to discard the changes in an active snapshot, the _remove_
operation will release any resources associated with the snapshot.  The mounts
provided by _prepare_ or _view_ should be unmounted before calling this method.

If the user wants to discard committed snapshots, the _remove_ operation can
also be used, but any children must be removed before proceeding.

For detailed usage information, see the
[GoDoc](https://godoc.org/github.com/containerd/containerd/snapshot#Snapshotter).

### Graph metadata

As snapshots are imported into the container system, a "graph" of snapshots and
their parents will form. Queries over this graph must be a supported operation.

## How snapshots work

To flesh out the _Snapshots_ terminology, we are going to demonstrate the use of
the _Snapshotter_ from the perspective of importing layers. We'll use a Go API
to represent the process.

### Importing a Layer

To import a layer, we simply have the _Snapshotter_ provide a list of
mounts to be applied such that our destination will capture a changeset. We start
out by getting a path to the layer tar file and creating a temp location to
unpack it to:

	layerPath, tmpDir := getLayerPath(), mkTmpDir() // just a path to layer tar file.

We start by using a _Snapshotter_ to _Prepare_ a new snapshot transaction, using
a _key_ and descending from the empty parent "":

	mounts, err := snapshotter.Prepare(key, "")
	if err != nil { ... }

We get back a list of mounts from `Snapshotter.Prepare`, with the `key`
identifying the active snapshot. Mount this to the temporary location with the
following:

	if err := mount.All(mounts, tmpDir); err != nil { ... }

Once the mounts are performed, our temporary location is ready to capture
a diff. In practice, this works similar to a filesystem transaction. The
next step is to unpack the layer. We have a special function `unpackLayer`
that applies the contents of the layer to target location and calculates the
`DiffID` of the unpacked layer (this is a requirement for docker
implementation):

	layer, err := os.Open(layerPath)
	if err != nil { ... }
	digest, err := unpackLayer(tmpLocation, layer) // unpack into layer location
	if err != nil { ... }

When the above completes, we should have a filesystem the represents the
contents of the layer. Careful implementations should verify that digest
matches the expected `DiffID`. When completed, we unmount the mounts:

	unmount(mounts) // optional, for now

Now that we've verified and unpacked our layer, we commit the active
snapshot to a _name_. For this example, we are just going to use the layer
digest, but in practice, this will probably be the `ChainID`:

	if err := snapshotter.Commit(digest.String(), key); err != nil { ... }

Now, we have a layer in the _Snapshotter_ that can be accessed with the digest
provided during commit. Once you have committed the snapshot, the active
snapshot can be removed with the following:

	snapshotter.Remove(key)

### Importing the Next Layer

Making a layer depend on the above is identical to the process described
above except that the parent is provided as `parent` when calling
`Snapshotter.Prepare`, assuming a clean `tmpLocation`:

	mounts, err := snapshotter.Prepare(tmpLocation, parentDigest)

We then mount, apply and commit, as we did above. The new snapshot will be
based on the content of the previous one.

### Running a Container

To run a container, we simply provide `Snapshotter.Prepare` the committed image
snapshot as the parent. After mounting, the prepared path can
be used directly as the container's filesystem:

	mounts, err := snapshotter.Prepare(containerKey, imageRootFSChainID)

The returned mounts can then be passed directly to the container runtime. If
one would like to create a new image from the filesystem, `Snapshotter.Commit`
is called:

	if err := snapshotter.Commit(newImageSnapshot, containerKey); err != nil { ... }

Alternatively, for most container runs, `Snapshotter.Remove` will be called to
signal the Snapshotter to abandon the changes.
