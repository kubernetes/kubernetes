package rootfs

import (
	"context"
	"io"
	"io/ioutil"

	"github.com/docker/containerd"
	"github.com/docker/containerd/archive"
	"github.com/docker/containerd/archive/compression"
	"github.com/docker/containerd/log"
	"github.com/docker/containerd/snapshot"
	"github.com/opencontainers/go-digest"
	"github.com/opencontainers/image-spec/identity"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/pkg/errors"
)

type Unpacker interface {
	Unpack(ctx context.Context, layers []ocispec.Descriptor) (digest.Digest, error)
}

type Mounter interface {
	Mount(target string, mounts ...containerd.Mount) error
	Unmount(target string) error
}

// ApplyLayer applies the layer to the provided parent. The resulting snapshot
// will be stored under its ChainID.
//
// The parent *must* be the chainID of the parent layer.
//
// The returned digest is the diffID for the applied layer.
func ApplyLayer(snapshots snapshot.Snapshotter, mounter Mounter, rd io.Reader, parent digest.Digest) (digest.Digest, error) {
	ctx := context.TODO()

	// create a temporary directory to work from, needs to be on same
	// filesystem. Probably better if this shared but we'll use a tempdir, for
	// now.
	dir, err := ioutil.TempDir("", "unpack-")
	if err != nil {
		return "", errors.Wrapf(err, "creating temporary directory failed")
	}

	// TODO(stevvooe): Choose this key WAY more carefully. We should be able to
	// create collisions for concurrent, conflicting unpack processes but we
	// would need to have it be a function of the parent diffID and child
	// layerID (since we don't know the diffID until we are done!).
	key := dir

	mounts, err := snapshots.Prepare(ctx, key, parent.String())
	if err != nil {
		return "", err
	}

	if err := mounter.Mount(dir, mounts...); err != nil {
		if err := snapshots.Remove(ctx, key); err != nil {
			log.L.WithError(err).Error("snapshot rollback failed")
		}
		return "", err
	}
	defer mounter.Unmount(dir)

	rd, err = compression.DecompressStream(rd)
	if err != nil {
		return "", err
	}

	digester := digest.Canonical.Digester() // used to calculate diffID.
	rd = io.TeeReader(rd, digester.Hash())

	if _, err := archive.Apply(context.Background(), key, rd); err != nil {
		return "", err
	}

	diffID := digester.Digest()

	chainID := diffID
	if parent != "" {
		chainID = identity.ChainID([]digest.Digest{parent, chainID})
	}
	if _, err := snapshots.Stat(ctx, chainID.String()); err == nil {
		return diffID, nil //TODO: call snapshots.Remove(ctx, key) once implemented
	}

	return diffID, snapshots.Commit(ctx, chainID.String(), key)
}

// Prepare the root filesystem from the set of layers. Snapshots are created
// for each layer if they don't exist, keyed by their chain id. If the snapshot
// already exists, it will be skipped.
//
// If successful, the chainID for the top-level layer is returned. That
// identifier can be used to check out a snapshot.
func Prepare(ctx context.Context, snapshots snapshot.Snapshotter, mounter Mounter, layers []ocispec.Descriptor,
	// TODO(stevvooe): The following functions are candidate for internal
	// object functions. We can use these to formulate the beginnings of a
	// rootfs Controller.
	//
	// Just pass them in for now.
	openBlob func(context.Context, digest.Digest) (io.ReadCloser, error),
	resolveDiffID func(digest.Digest) digest.Digest,
	registerDiffID func(diffID, dgst digest.Digest) error) (digest.Digest, error) {
	var (
		parent digest.Digest
		chain  []digest.Digest
	)

	for _, layer := range layers {
		// TODO: layer.Digest should not be string
		// (https://github.com/opencontainers/image-spec/pull/514)
		layerDigest := digest.Digest(layer.Digest)
		// This will convert a possibly compressed layer hash to the
		// uncompressed hash, if we know about it. If we don't, we unpack and
		// calculate it. If we do have it, we then calculate the chain id for
		// the application and see if the snapshot is there.
		diffID := resolveDiffID(layerDigest)
		if diffID != "" {
			chainLocal := append(chain, diffID)
			chainID := identity.ChainID(chainLocal)

			if _, err := snapshots.Stat(ctx, chainID.String()); err == nil {
				continue
			}
		}

		rc, err := openBlob(ctx, layerDigest)
		if err != nil {
			return "", err
		}
		defer rc.Close() // pretty lazy!

		diffID, err = ApplyLayer(snapshots, mounter, rc, parent)
		if err != nil {
			return "", err
		}

		// Register the association between the diffID and the layer's digest.
		// For uncompressed layers, this will be the same. For compressed
		// layers, we can look up the diffID from the digest if we've already
		// unpacked it.
		if err := registerDiffID(diffID, layerDigest); err != nil {
			return "", err
		}

		chain = append(chain, diffID)
		parent = identity.ChainID(chain)
	}

	return parent, nil
}
