package layer

import (
	"fmt"
	"io"

	"github.com/docker/distribution"
	"github.com/opencontainers/go-digest"
)

type roLayer struct {
	chainID    ChainID
	diffID     DiffID
	parent     *roLayer
	cacheID    string
	size       int64
	layerStore *layerStore
	descriptor distribution.Descriptor
	platform   Platform

	referenceCount int
	references     map[Layer]struct{}
}

// TarStream for roLayer guarantees that the data that is produced is the exact
// data that the layer was registered with.
func (rl *roLayer) TarStream() (io.ReadCloser, error) {
	rc, err := rl.layerStore.getTarStream(rl)
	if err != nil {
		return nil, err
	}

	vrc, err := newVerifiedReadCloser(rc, digest.Digest(rl.diffID))
	if err != nil {
		return nil, err
	}
	return vrc, nil
}

// TarStreamFrom does not make any guarantees to the correctness of the produced
// data. As such it should not be used when the layer content must be verified
// to be an exact match to the registered layer.
func (rl *roLayer) TarStreamFrom(parent ChainID) (io.ReadCloser, error) {
	var parentCacheID string
	for pl := rl.parent; pl != nil; pl = pl.parent {
		if pl.chainID == parent {
			parentCacheID = pl.cacheID
			break
		}
	}

	if parent != ChainID("") && parentCacheID == "" {
		return nil, fmt.Errorf("layer ID '%s' is not a parent of the specified layer: cannot provide diff to non-parent", parent)
	}
	return rl.layerStore.driver.Diff(rl.cacheID, parentCacheID)
}

func (rl *roLayer) ChainID() ChainID {
	return rl.chainID
}

func (rl *roLayer) DiffID() DiffID {
	return rl.diffID
}

func (rl *roLayer) Parent() Layer {
	if rl.parent == nil {
		return nil
	}
	return rl.parent
}

func (rl *roLayer) Size() (size int64, err error) {
	if rl.parent != nil {
		size, err = rl.parent.Size()
		if err != nil {
			return
		}
	}

	return size + rl.size, nil
}

func (rl *roLayer) DiffSize() (size int64, err error) {
	return rl.size, nil
}

func (rl *roLayer) Metadata() (map[string]string, error) {
	return rl.layerStore.driver.GetMetadata(rl.cacheID)
}

type referencedCacheLayer struct {
	*roLayer
}

func (rl *roLayer) getReference() Layer {
	ref := &referencedCacheLayer{
		roLayer: rl,
	}
	rl.references[ref] = struct{}{}

	return ref
}

func (rl *roLayer) hasReference(ref Layer) bool {
	_, ok := rl.references[ref]
	return ok
}

func (rl *roLayer) hasReferences() bool {
	return len(rl.references) > 0
}

func (rl *roLayer) deleteReference(ref Layer) {
	delete(rl.references, ref)
}

func (rl *roLayer) depth() int {
	if rl.parent == nil {
		return 1
	}
	return rl.parent.depth() + 1
}

func storeLayer(tx MetadataTransaction, layer *roLayer) error {
	if err := tx.SetDiffID(layer.diffID); err != nil {
		return err
	}
	if err := tx.SetSize(layer.size); err != nil {
		return err
	}
	if err := tx.SetCacheID(layer.cacheID); err != nil {
		return err
	}
	// Do not store empty descriptors
	if layer.descriptor.Digest != "" {
		if err := tx.SetDescriptor(layer.descriptor); err != nil {
			return err
		}
	}
	if layer.parent != nil {
		if err := tx.SetParent(layer.parent.chainID); err != nil {
			return err
		}
	}
	if err := tx.SetPlatform(layer.platform); err != nil {
		return err
	}

	return nil
}

func newVerifiedReadCloser(rc io.ReadCloser, dgst digest.Digest) (io.ReadCloser, error) {
	return &verifiedReadCloser{
		rc:       rc,
		dgst:     dgst,
		verifier: dgst.Verifier(),
	}, nil
}

type verifiedReadCloser struct {
	rc       io.ReadCloser
	dgst     digest.Digest
	verifier digest.Verifier
}

func (vrc *verifiedReadCloser) Read(p []byte) (n int, err error) {
	n, err = vrc.rc.Read(p)
	if n > 0 {
		if n, err := vrc.verifier.Write(p[:n]); err != nil {
			return n, err
		}
	}
	if err == io.EOF {
		if !vrc.verifier.Verified() {
			err = fmt.Errorf("could not verify layer data for: %s. This may be because internal files in the layer store were modified. Re-pulling or rebuilding this image may resolve the issue", vrc.dgst)
		}
	}
	return
}
func (vrc *verifiedReadCloser) Close() error {
	return vrc.rc.Close()
}
