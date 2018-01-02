package snapshot

import (
	"context"
	"encoding/json"
	"strings"
	"time"

	"github.com/containerd/containerd/mount"
)

// Kind identifies the kind of snapshot.
type Kind uint8

// definitions of snapshot kinds
const (
	KindUnknown Kind = iota
	KindView
	KindActive
	KindCommitted
)

// ParseKind parses the provided string into a Kind
//
// If the string cannot be parsed KindUnknown is returned
func ParseKind(s string) Kind {
	s = strings.ToLower(s)
	switch s {
	case "view":
		return KindView
	case "active":
		return KindActive
	case "committed":
		return KindCommitted
	}

	return KindUnknown
}

// String returns the string representation of the Kind
func (k Kind) String() string {
	switch k {
	case KindView:
		return "View"
	case KindActive:
		return "Active"
	case KindCommitted:
		return "Committed"
	}

	return "Unknown"
}

// MarshalJSON the Kind to JSON
func (k Kind) MarshalJSON() ([]byte, error) {
	return json.Marshal(k.String())
}

// UnmarshalJSON the Kind from JSON
func (k *Kind) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}

	*k = ParseKind(s)
	return nil
}

// Info provides information about a particular snapshot.
// JSON marshallability is supported for interactive with tools like ctr,
type Info struct {
	Kind    Kind              // active or committed snapshot
	Name    string            // name or key of snapshot
	Parent  string            `json:",omitempty"` // name of parent snapshot
	Labels  map[string]string `json:",omitempty"` // Labels for snapshot
	Created time.Time         `json:",omitempty"` // Created time
	Updated time.Time         `json:",omitempty"` // Last update time
}

// Usage defines statistics for disk resources consumed by the snapshot.
//
// These resources only include the resources consumed by the snapshot itself
// and does not include resources usage by the parent.
type Usage struct {
	Inodes int64 // number of inodes in use.
	Size   int64 // provides usage, in bytes, of snapshot
}

// Add the provided usage to the current usage
func (u *Usage) Add(other Usage) {
	u.Size += other.Size

	// TODO(stevvooe): assumes independent inodes, but provides and upper
	// bound. This should be pretty close, assuming the inodes for a
	// snapshot are roughly unique to it. Don't trust this assumption.
	u.Inodes += other.Inodes
}

// Snapshotter defines the methods required to implement a snapshot snapshotter for
// allocating, snapshotting and mounting filesystem changesets. The model works
// by building up sets of changes with parent-child relationships.
//
// A snapshot represents a filesystem state. Every snapshot has a parent, where
// the empty parent is represented by the empty string. A diff can be taken
// between a parent and its snapshot to generate a classic layer.
//
// An active snapshot is created by calling `Prepare`. After mounting, changes
// can be made to the snapshot. The act of committing creates a committed
// snapshot. The committed snapshot will get the parent of active snapshot. The
// committed snapshot can then be used as a parent. Active snapshots can never
// act as a parent.
//
// Snapshots are best understood by their lifecycle. Active snapshots are
// always created with Prepare or View. Committed snapshots are always created
// with Commit.  Active snapshots never become committed snapshots and vice
// versa. All snapshots may be removed.
//
// For consistency, we define the following terms to be used throughout this
// interface for snapshotter implementations:
//
// 	`ctx` - refers to a context.Context
// 	`key` - refers to an active snapshot
// 	`name` - refers to a committed snapshot
// 	`parent` - refers to the parent in relation
//
// Most methods take various combinations of these identifiers. Typically,
// `name` and `parent` will be used in cases where a method *only* takes
// committed snapshots. `key` will be used to refer to active snapshots in most
// cases, except where noted. All variables used to access snapshots use the
// same key space. For example, an active snapshot may not share the same key
// with a committed snapshot.
//
// We cover several examples below to demonstrate the utility of a snapshot
// snapshotter.
//
// Importing a Layer
//
// To import a layer, we simply have the Snapshotter provide a list of
// mounts to be applied such that our dst will capture a changeset. We start
// out by getting a path to the layer tar file and creating a temp location to
// unpack it to:
//
//	layerPath, tmpDir := getLayerPath(), mkTmpDir() // just a path to layer tar file.
//
// We start by using a Snapshotter to Prepare a new snapshot transaction, using a
// key and descending from the empty parent "":
//
//	mounts, err := snapshotter.Prepare(ctx, key, "")
// 	if err != nil { ... }
//
// We get back a list of mounts from Snapshotter.Prepare, with the key identifying
// the active snapshot. Mount this to the temporary location with the
// following:
//
//	if err := mount.All(mounts, tmpDir); err != nil { ... }
//
// Once the mounts are performed, our temporary location is ready to capture
// a diff. In practice, this works similar to a filesystem transaction. The
// next step is to unpack the layer. We have a special function unpackLayer
// that applies the contents of the layer to target location and calculates the
// DiffID of the unpacked layer (this is a requirement for docker
// implementation):
//
//	layer, err := os.Open(layerPath)
//	if err != nil { ... }
// 	digest, err := unpackLayer(tmpLocation, layer) // unpack into layer location
// 	if err != nil { ... }
//
// When the above completes, we should have a filesystem the represents the
// contents of the layer. Careful implementations should verify that digest
// matches the expected DiffID. When completed, we unmount the mounts:
//
//	unmount(mounts) // optional, for now
//
// Now that we've verified and unpacked our layer, we commit the active
// snapshot to a name. For this example, we are just going to use the layer
// digest, but in practice, this will probably be the ChainID:
//
//	if err := snapshotter.Commit(ctx, digest.String(), key); err != nil { ... }
//
// Now, we have a layer in the Snapshotter that can be accessed with the digest
// provided during commit. Once you have committed the snapshot, the active
// snapshot can be removed with the following:
//
// 	snapshotter.Remove(ctx, key)
//
// Importing the Next Layer
//
// Making a layer depend on the above is identical to the process described
// above except that the parent is provided as parent when calling
// Manager.Prepare, assuming a clean, unique key identifier:
//
// 	mounts, err := snapshotter.Prepare(ctx, key, parentDigest)
//
// We then mount, apply and commit, as we did above. The new snapshot will be
// based on the content of the previous one.
//
// Running a Container
//
// To run a container, we simply provide Snapshotter.Prepare the committed image
// snapshot as the parent. After mounting, the prepared path can
// be used directly as the container's filesystem:
//
// 	mounts, err := snapshotter.Prepare(ctx, containerKey, imageRootFSChainID)
//
// The returned mounts can then be passed directly to the container runtime. If
// one would like to create a new image from the filesystem, Manager.Commit is
// called:
//
// 	if err := snapshotter.Commit(ctx, newImageSnapshot, containerKey); err != nil { ... }
//
// Alternatively, for most container runs, Snapshotter.Remove will be called to
// signal the Snapshotter to abandon the changes.
type Snapshotter interface {
	// Stat returns the info for an active or committed snapshot by name or
	// key.
	//
	// Should be used for parent resolution, existence checks and to discern
	// the kind of snapshot.
	Stat(ctx context.Context, key string) (Info, error)

	// Update updates the info for a snapshot.
	//
	// Only mutable properties of a snapshot may be updated.
	Update(ctx context.Context, info Info, fieldpaths ...string) (Info, error)

	// Usage returns the resource usage of an active or committed snapshot
	// excluding the usage of parent snapshots.
	//
	// The running time of this call for active snapshots is dependent on
	// implementation, but may be proportional to the size of the resource.
	// Callers should take this into consideration. Implementations should
	// attempt to honer context cancellation and avoid taking locks when making
	// the calculation.
	Usage(ctx context.Context, key string) (Usage, error)

	// Mounts returns the mounts for the active snapshot transaction identified
	// by key. Can be called on an read-write or readonly transaction. This is
	// available only for active snapshots.
	//
	// This can be used to recover mounts after calling View or Prepare.
	Mounts(ctx context.Context, key string) ([]mount.Mount, error)

	// Prepare creates an active snapshot identified by key descending from the
	// provided parent.  The returned mounts can be used to mount the snapshot
	// to capture changes.
	//
	// If a parent is provided, after performing the mounts, the destination
	// will start with the content of the parent. The parent must be a
	// committed snapshot. Changes to the mounted destination will be captured
	// in relation to the parent. The default parent, "", is an empty
	// directory.
	//
	// The changes may be saved to a committed snapshot by calling Commit. When
	// one is done with the transaction, Remove should be called on the key.
	//
	// Multiple calls to Prepare or View with the same key should fail.
	Prepare(ctx context.Context, key, parent string, opts ...Opt) ([]mount.Mount, error)

	// View behaves identically to Prepare except the result may not be
	// committed back to the snapshot snapshotter. View returns a readonly view on
	// the parent, with the active snapshot being tracked by the given key.
	//
	// This method operates identically to Prepare, except that Mounts returned
	// may have the readonly flag set. Any modifications to the underlying
	// filesystem will be ignored. Implementations may perform this in a more
	// efficient manner that differs from what would be attempted with
	// `Prepare`.
	//
	// Commit may not be called on the provided key and will return an error.
	// To collect the resources associated with key, Remove must be called with
	// key as the argument.
	View(ctx context.Context, key, parent string, opts ...Opt) ([]mount.Mount, error)

	// Commit captures the changes between key and its parent into a snapshot
	// identified by name.  The name can then be used with the snapshotter's other
	// methods to create subsequent snapshots.
	//
	// A committed snapshot will be created under name with the parent of the
	// active snapshot.
	//
	// Commit may be called multiple times on the same key. Snapshots created
	// in this manner will all reference the parent used to start the
	// transaction.
	Commit(ctx context.Context, name, key string, opts ...Opt) error

	// Remove the committed or active snapshot by the provided key.
	//
	// All resources associated with the key will be removed.
	//
	// If the snapshot is a parent of another snapshot, its children must be
	// removed before proceeding.
	Remove(ctx context.Context, key string) error

	// Walk all snapshots in the snapshotter. For each snapshot in the
	// snapshotter, the function will be called.
	Walk(ctx context.Context, fn func(context.Context, Info) error) error
}

// Opt allows setting mutable snapshot properties on creation
type Opt func(info *Info) error

// WithLabels adds labels to a created snapshot
func WithLabels(labels map[string]string) Opt {
	return func(info *Info) error {
		info.Labels = labels
		return nil
	}
}
