package fscache

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/moby/buildkit/session/filesync"
	"github.com/stretchr/testify/assert"
	"golang.org/x/net/context"
)

func TestFSCache(t *testing.T) {
	tmpDir, err := ioutil.TempDir("", "fscache")
	assert.Nil(t, err)
	defer os.RemoveAll(tmpDir)

	backend := NewNaiveCacheBackend(filepath.Join(tmpDir, "backend"))

	opt := Opt{
		Root:     tmpDir,
		Backend:  backend,
		GCPolicy: GCPolicy{MaxSize: 15, MaxKeepDuration: time.Hour},
	}

	fscache, err := NewFSCache(opt)
	assert.Nil(t, err)

	defer fscache.Close()

	err = fscache.RegisterTransport("test", &testTransport{})
	assert.Nil(t, err)

	src1, err := fscache.SyncFrom(context.TODO(), &testIdentifier{"foo", "data", "bar"})
	assert.Nil(t, err)

	dt, err := ioutil.ReadFile(filepath.Join(src1.Root(), "foo"))
	assert.Nil(t, err)
	assert.Equal(t, string(dt), "data")

	// same id doesn't recalculate anything
	src2, err := fscache.SyncFrom(context.TODO(), &testIdentifier{"foo", "data2", "bar"})
	assert.Nil(t, err)
	assert.Equal(t, src1.Root(), src2.Root())

	dt, err = ioutil.ReadFile(filepath.Join(src1.Root(), "foo"))
	assert.Nil(t, err)
	assert.Equal(t, string(dt), "data")
	assert.Nil(t, src2.Close())

	src3, err := fscache.SyncFrom(context.TODO(), &testIdentifier{"foo2", "data2", "bar"})
	assert.Nil(t, err)
	assert.NotEqual(t, src1.Root(), src3.Root())

	dt, err = ioutil.ReadFile(filepath.Join(src3.Root(), "foo2"))
	assert.Nil(t, err)
	assert.Equal(t, string(dt), "data2")

	s, err := fscache.DiskUsage()
	assert.Nil(t, err)
	assert.Equal(t, s, int64(0))

	assert.Nil(t, src3.Close())

	s, err = fscache.DiskUsage()
	assert.Nil(t, err)
	assert.Equal(t, s, int64(5))

	// new upload with the same shared key shoutl overwrite
	src4, err := fscache.SyncFrom(context.TODO(), &testIdentifier{"foo3", "data3", "bar"})
	assert.Nil(t, err)
	assert.NotEqual(t, src1.Root(), src3.Root())

	dt, err = ioutil.ReadFile(filepath.Join(src3.Root(), "foo3"))
	assert.Nil(t, err)
	assert.Equal(t, string(dt), "data3")
	assert.Equal(t, src4.Root(), src3.Root())
	assert.Nil(t, src4.Close())

	s, err = fscache.DiskUsage()
	assert.Nil(t, err)
	assert.Equal(t, s, int64(10))

	// this one goes over the GC limit
	src5, err := fscache.SyncFrom(context.TODO(), &testIdentifier{"foo4", "datadata", "baz"})
	assert.Nil(t, err)
	assert.Nil(t, src5.Close())

	// GC happens async
	time.Sleep(100 * time.Millisecond)

	// only last insertion after GC
	s, err = fscache.DiskUsage()
	assert.Nil(t, err)
	assert.Equal(t, s, int64(8))

	// prune deletes everything
	released, err := fscache.Prune(context.TODO())
	assert.Nil(t, err)
	assert.Equal(t, released, uint64(8))

	s, err = fscache.DiskUsage()
	assert.Nil(t, err)
	assert.Equal(t, s, int64(0))
}

type testTransport struct {
}

func (t *testTransport) Copy(ctx context.Context, id RemoteIdentifier, dest string, cs filesync.CacheUpdater) error {
	testid := id.(*testIdentifier)
	return ioutil.WriteFile(filepath.Join(dest, testid.filename), []byte(testid.data), 0600)
}

type testIdentifier struct {
	filename  string
	data      string
	sharedKey string
}

func (t *testIdentifier) Key() string {
	return t.filename
}
func (t *testIdentifier) SharedKey() string {
	return t.sharedKey
}
func (t *testIdentifier) Transport() string {
	return "test"
}
