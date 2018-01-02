package image

import (
	"runtime"
	"testing"

	"github.com/docker/docker/layer"
	"github.com/docker/docker/pkg/testutil"
	"github.com/opencontainers/go-digest"
	"github.com/stretchr/testify/assert"
)

func TestRestore(t *testing.T) {
	fs, cleanup := defaultFSStoreBackend(t)
	defer cleanup()

	id1, err := fs.Set([]byte(`{"comment": "abc", "rootfs": {"type": "layers"}}`))
	assert.NoError(t, err)

	_, err = fs.Set([]byte(`invalid`))
	assert.NoError(t, err)

	id2, err := fs.Set([]byte(`{"comment": "def", "rootfs": {"type": "layers", "diff_ids": ["2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae"]}}`))
	assert.NoError(t, err)

	err = fs.SetMetadata(id2, "parent", []byte(id1))
	assert.NoError(t, err)

	is, err := NewImageStore(fs, runtime.GOOS, &mockLayerGetReleaser{})
	assert.NoError(t, err)

	assert.Len(t, is.Map(), 2)

	img1, err := is.Get(ID(id1))
	assert.NoError(t, err)
	assert.Equal(t, ID(id1), img1.computedID)
	assert.Equal(t, string(id1), img1.computedID.String())

	img2, err := is.Get(ID(id2))
	assert.NoError(t, err)
	assert.Equal(t, "abc", img1.Comment)
	assert.Equal(t, "def", img2.Comment)

	p, err := is.GetParent(ID(id1))
	testutil.ErrorContains(t, err, "failed to read metadata")

	p, err = is.GetParent(ID(id2))
	assert.NoError(t, err)
	assert.Equal(t, ID(id1), p)

	children := is.Children(ID(id1))
	assert.Len(t, children, 1)
	assert.Equal(t, ID(id2), children[0])
	assert.Len(t, is.Heads(), 1)

	sid1, err := is.Search(string(id1)[:10])
	assert.NoError(t, err)
	assert.Equal(t, ID(id1), sid1)

	sid1, err = is.Search(digest.Digest(id1).Hex()[:6])
	assert.NoError(t, err)
	assert.Equal(t, ID(id1), sid1)

	invalidPattern := digest.Digest(id1).Hex()[1:6]
	_, err = is.Search(invalidPattern)
	testutil.ErrorContains(t, err, "No such image")
}

func TestAddDelete(t *testing.T) {
	is, cleanup := defaultImageStore(t)
	defer cleanup()

	id1, err := is.Create([]byte(`{"comment": "abc", "rootfs": {"type": "layers", "diff_ids": ["2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae"]}}`))
	assert.NoError(t, err)
	assert.Equal(t, ID("sha256:8d25a9c45df515f9d0fe8e4a6b1c64dd3b965a84790ddbcc7954bb9bc89eb993"), id1)

	img, err := is.Get(id1)
	assert.NoError(t, err)
	assert.Equal(t, "abc", img.Comment)

	id2, err := is.Create([]byte(`{"comment": "def", "rootfs": {"type": "layers", "diff_ids": ["2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae"]}}`))
	assert.NoError(t, err)

	err = is.SetParent(id2, id1)
	assert.NoError(t, err)

	pid1, err := is.GetParent(id2)
	assert.NoError(t, err)
	assert.Equal(t, pid1, id1)

	_, err = is.Delete(id1)
	assert.NoError(t, err)

	_, err = is.Get(id1)
	testutil.ErrorContains(t, err, "failed to get digest")

	_, err = is.Get(id2)
	assert.NoError(t, err)

	_, err = is.GetParent(id2)
	testutil.ErrorContains(t, err, "failed to read metadata")
}

func TestSearchAfterDelete(t *testing.T) {
	is, cleanup := defaultImageStore(t)
	defer cleanup()

	id, err := is.Create([]byte(`{"comment": "abc", "rootfs": {"type": "layers"}}`))
	assert.NoError(t, err)

	id1, err := is.Search(string(id)[:15])
	assert.NoError(t, err)
	assert.Equal(t, id1, id)

	_, err = is.Delete(id)
	assert.NoError(t, err)

	_, err = is.Search(string(id)[:15])
	testutil.ErrorContains(t, err, "No such image")
}

func TestParentReset(t *testing.T) {
	is, cleanup := defaultImageStore(t)
	defer cleanup()

	id, err := is.Create([]byte(`{"comment": "abc1", "rootfs": {"type": "layers"}}`))
	assert.NoError(t, err)

	id2, err := is.Create([]byte(`{"comment": "abc2", "rootfs": {"type": "layers"}}`))
	assert.NoError(t, err)

	id3, err := is.Create([]byte(`{"comment": "abc3", "rootfs": {"type": "layers"}}`))
	assert.NoError(t, err)

	assert.NoError(t, is.SetParent(id, id2))
	assert.Len(t, is.Children(id2), 1)

	assert.NoError(t, is.SetParent(id, id3))
	assert.Len(t, is.Children(id2), 0)
	assert.Len(t, is.Children(id3), 1)
}

func defaultImageStore(t *testing.T) (Store, func()) {
	fsBackend, cleanup := defaultFSStoreBackend(t)

	store, err := NewImageStore(fsBackend, runtime.GOOS, &mockLayerGetReleaser{})
	assert.NoError(t, err)

	return store, cleanup
}

func TestGetAndSetLastUpdated(t *testing.T) {
	store, cleanup := defaultImageStore(t)
	defer cleanup()

	id, err := store.Create([]byte(`{"comment": "abc1", "rootfs": {"type": "layers"}}`))
	assert.NoError(t, err)

	updated, err := store.GetLastUpdated(id)
	assert.NoError(t, err)
	assert.Equal(t, updated.IsZero(), true)

	assert.NoError(t, store.SetLastUpdated(id))

	updated, err = store.GetLastUpdated(id)
	assert.NoError(t, err)
	assert.Equal(t, updated.IsZero(), false)
}

type mockLayerGetReleaser struct{}

func (ls *mockLayerGetReleaser) Get(layer.ChainID) (layer.Layer, error) {
	return nil, nil
}

func (ls *mockLayerGetReleaser) Release(layer.Layer) ([]layer.Metadata, error) {
	return nil, nil
}
