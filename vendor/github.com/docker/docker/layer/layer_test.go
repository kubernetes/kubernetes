package layer

import (
	"bytes"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/docker/docker/daemon/graphdriver"
	"github.com/docker/docker/daemon/graphdriver/vfs"
	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/idtools"
	"github.com/docker/docker/pkg/stringid"
	"github.com/opencontainers/go-digest"
)

func init() {
	graphdriver.ApplyUncompressedLayer = archive.UnpackLayer
	defaultArchiver := archive.NewDefaultArchiver()
	vfs.CopyWithTar = defaultArchiver.CopyWithTar
}

func newVFSGraphDriver(td string) (graphdriver.Driver, error) {
	uidMap := []idtools.IDMap{
		{
			ContainerID: 0,
			HostID:      os.Getuid(),
			Size:        1,
		},
	}
	gidMap := []idtools.IDMap{
		{
			ContainerID: 0,
			HostID:      os.Getgid(),
			Size:        1,
		},
	}

	options := graphdriver.Options{Root: td, UIDMaps: uidMap, GIDMaps: gidMap}
	return graphdriver.GetDriver("vfs", nil, options)
}

func newTestGraphDriver(t *testing.T) (graphdriver.Driver, func()) {
	td, err := ioutil.TempDir("", "graph-")
	if err != nil {
		t.Fatal(err)
	}

	driver, err := newVFSGraphDriver(td)
	if err != nil {
		t.Fatal(err)
	}

	return driver, func() {
		os.RemoveAll(td)
	}
}

func newTestStore(t *testing.T) (Store, string, func()) {
	td, err := ioutil.TempDir("", "layerstore-")
	if err != nil {
		t.Fatal(err)
	}

	graph, graphcleanup := newTestGraphDriver(t)
	fms, err := NewFSMetadataStore(td)
	if err != nil {
		t.Fatal(err)
	}
	ls, err := NewStoreFromGraphDriver(fms, graph, runtime.GOOS)
	if err != nil {
		t.Fatal(err)
	}

	return ls, td, func() {
		graphcleanup()
		os.RemoveAll(td)
	}
}

type layerInit func(root string) error

func createLayer(ls Store, parent ChainID, layerFunc layerInit) (Layer, error) {
	containerID := stringid.GenerateRandomID()
	mount, err := ls.CreateRWLayer(containerID, parent, nil)
	if err != nil {
		return nil, err
	}

	path, err := mount.Mount("")
	if err != nil {
		return nil, err
	}

	if err := layerFunc(path); err != nil {
		return nil, err
	}

	ts, err := mount.TarStream()
	if err != nil {
		return nil, err
	}
	defer ts.Close()

	layer, err := ls.Register(ts, parent, Platform(runtime.GOOS))
	if err != nil {
		return nil, err
	}

	if err := mount.Unmount(); err != nil {
		return nil, err
	}

	if _, err := ls.ReleaseRWLayer(mount); err != nil {
		return nil, err
	}

	return layer, nil
}

type FileApplier interface {
	ApplyFile(root string) error
}

type testFile struct {
	name       string
	content    []byte
	permission os.FileMode
}

func newTestFile(name string, content []byte, perm os.FileMode) FileApplier {
	return &testFile{
		name:       name,
		content:    content,
		permission: perm,
	}
}

func (tf *testFile) ApplyFile(root string) error {
	fullPath := filepath.Join(root, tf.name)
	if err := os.MkdirAll(filepath.Dir(fullPath), 0755); err != nil {
		return err
	}
	// Check if already exists
	if stat, err := os.Stat(fullPath); err == nil && stat.Mode().Perm() != tf.permission {
		if err := os.Chmod(fullPath, tf.permission); err != nil {
			return err
		}
	}
	if err := ioutil.WriteFile(fullPath, tf.content, tf.permission); err != nil {
		return err
	}
	return nil
}

func initWithFiles(files ...FileApplier) layerInit {
	return func(root string) error {
		for _, f := range files {
			if err := f.ApplyFile(root); err != nil {
				return err
			}
		}
		return nil
	}
}

func getCachedLayer(l Layer) *roLayer {
	if rl, ok := l.(*referencedCacheLayer); ok {
		return rl.roLayer
	}
	return l.(*roLayer)
}

func getMountLayer(l RWLayer) *mountedLayer {
	return l.(*referencedRWLayer).mountedLayer
}

func createMetadata(layers ...Layer) []Metadata {
	metadata := make([]Metadata, len(layers))
	for i := range layers {
		size, err := layers[i].Size()
		if err != nil {
			panic(err)
		}

		metadata[i].ChainID = layers[i].ChainID()
		metadata[i].DiffID = layers[i].DiffID()
		metadata[i].Size = size
		metadata[i].DiffSize = getCachedLayer(layers[i]).size
	}

	return metadata
}

func assertMetadata(t *testing.T, metadata, expectedMetadata []Metadata) {
	if len(metadata) != len(expectedMetadata) {
		t.Fatalf("Unexpected number of deletes %d, expected %d", len(metadata), len(expectedMetadata))
	}

	for i := range metadata {
		if metadata[i] != expectedMetadata[i] {
			t.Errorf("Unexpected metadata\n\tExpected: %#v\n\tActual: %#v", expectedMetadata[i], metadata[i])
		}
	}
	if t.Failed() {
		t.FailNow()
	}
}

func releaseAndCheckDeleted(t *testing.T, ls Store, layer Layer, removed ...Layer) {
	layerCount := len(ls.(*layerStore).layerMap)
	expectedMetadata := createMetadata(removed...)
	metadata, err := ls.Release(layer)
	if err != nil {
		t.Fatal(err)
	}

	assertMetadata(t, metadata, expectedMetadata)

	if expected := layerCount - len(removed); len(ls.(*layerStore).layerMap) != expected {
		t.Fatalf("Unexpected number of layers %d, expected %d", len(ls.(*layerStore).layerMap), expected)
	}
}

func cacheID(l Layer) string {
	return getCachedLayer(l).cacheID
}

func assertLayerEqual(t *testing.T, l1, l2 Layer) {
	if l1.ChainID() != l2.ChainID() {
		t.Fatalf("Mismatched ChainID: %s vs %s", l1.ChainID(), l2.ChainID())
	}
	if l1.DiffID() != l2.DiffID() {
		t.Fatalf("Mismatched DiffID: %s vs %s", l1.DiffID(), l2.DiffID())
	}

	size1, err := l1.Size()
	if err != nil {
		t.Fatal(err)
	}

	size2, err := l2.Size()
	if err != nil {
		t.Fatal(err)
	}

	if size1 != size2 {
		t.Fatalf("Mismatched size: %d vs %d", size1, size2)
	}

	if cacheID(l1) != cacheID(l2) {
		t.Fatalf("Mismatched cache id: %s vs %s", cacheID(l1), cacheID(l2))
	}

	p1 := l1.Parent()
	p2 := l2.Parent()
	if p1 != nil && p2 != nil {
		assertLayerEqual(t, p1, p2)
	} else if p1 != nil || p2 != nil {
		t.Fatalf("Mismatched parents: %v vs %v", p1, p2)
	}
}

func TestMountAndRegister(t *testing.T) {
	ls, _, cleanup := newTestStore(t)
	defer cleanup()

	li := initWithFiles(newTestFile("testfile.txt", []byte("some test data"), 0644))
	layer, err := createLayer(ls, "", li)
	if err != nil {
		t.Fatal(err)
	}

	size, _ := layer.Size()
	t.Logf("Layer size: %d", size)

	mount2, err := ls.CreateRWLayer("new-test-mount", layer.ChainID(), nil)
	if err != nil {
		t.Fatal(err)
	}

	path2, err := mount2.Mount("")
	if err != nil {
		t.Fatal(err)
	}

	b, err := ioutil.ReadFile(filepath.Join(path2, "testfile.txt"))
	if err != nil {
		t.Fatal(err)
	}

	if expected := "some test data"; string(b) != expected {
		t.Fatalf("Wrong file data, expected %q, got %q", expected, string(b))
	}

	if err := mount2.Unmount(); err != nil {
		t.Fatal(err)
	}

	if _, err := ls.ReleaseRWLayer(mount2); err != nil {
		t.Fatal(err)
	}
}

func TestLayerRelease(t *testing.T) {
	// TODO Windows: Figure out why this is failing
	if runtime.GOOS == "windows" {
		t.Skip("Failing on Windows")
	}
	ls, _, cleanup := newTestStore(t)
	defer cleanup()

	layer1, err := createLayer(ls, "", initWithFiles(newTestFile("layer1.txt", []byte("layer 1 file"), 0644)))
	if err != nil {
		t.Fatal(err)
	}

	layer2, err := createLayer(ls, layer1.ChainID(), initWithFiles(newTestFile("layer2.txt", []byte("layer 2 file"), 0644)))
	if err != nil {
		t.Fatal(err)
	}

	if _, err := ls.Release(layer1); err != nil {
		t.Fatal(err)
	}

	layer3a, err := createLayer(ls, layer2.ChainID(), initWithFiles(newTestFile("layer3.txt", []byte("layer 3a file"), 0644)))
	if err != nil {
		t.Fatal(err)
	}

	layer3b, err := createLayer(ls, layer2.ChainID(), initWithFiles(newTestFile("layer3.txt", []byte("layer 3b file"), 0644)))
	if err != nil {
		t.Fatal(err)
	}

	if _, err := ls.Release(layer2); err != nil {
		t.Fatal(err)
	}

	t.Logf("Layer1:  %s", layer1.ChainID())
	t.Logf("Layer2:  %s", layer2.ChainID())
	t.Logf("Layer3a: %s", layer3a.ChainID())
	t.Logf("Layer3b: %s", layer3b.ChainID())

	if expected := 4; len(ls.(*layerStore).layerMap) != expected {
		t.Fatalf("Unexpected number of layers %d, expected %d", len(ls.(*layerStore).layerMap), expected)
	}

	releaseAndCheckDeleted(t, ls, layer3b, layer3b)
	releaseAndCheckDeleted(t, ls, layer3a, layer3a, layer2, layer1)
}

func TestStoreRestore(t *testing.T) {
	// TODO Windows: Figure out why this is failing
	if runtime.GOOS == "windows" {
		t.Skip("Failing on Windows")
	}
	ls, _, cleanup := newTestStore(t)
	defer cleanup()

	layer1, err := createLayer(ls, "", initWithFiles(newTestFile("layer1.txt", []byte("layer 1 file"), 0644)))
	if err != nil {
		t.Fatal(err)
	}

	layer2, err := createLayer(ls, layer1.ChainID(), initWithFiles(newTestFile("layer2.txt", []byte("layer 2 file"), 0644)))
	if err != nil {
		t.Fatal(err)
	}

	if _, err := ls.Release(layer1); err != nil {
		t.Fatal(err)
	}

	layer3, err := createLayer(ls, layer2.ChainID(), initWithFiles(newTestFile("layer3.txt", []byte("layer 3 file"), 0644)))
	if err != nil {
		t.Fatal(err)
	}

	if _, err := ls.Release(layer2); err != nil {
		t.Fatal(err)
	}

	m, err := ls.CreateRWLayer("some-mount_name", layer3.ChainID(), nil)
	if err != nil {
		t.Fatal(err)
	}

	path, err := m.Mount("")
	if err != nil {
		t.Fatal(err)
	}

	if err := ioutil.WriteFile(filepath.Join(path, "testfile.txt"), []byte("nothing here"), 0644); err != nil {
		t.Fatal(err)
	}

	if err := m.Unmount(); err != nil {
		t.Fatal(err)
	}

	ls2, err := NewStoreFromGraphDriver(ls.(*layerStore).store, ls.(*layerStore).driver, runtime.GOOS)
	if err != nil {
		t.Fatal(err)
	}

	layer3b, err := ls2.Get(layer3.ChainID())
	if err != nil {
		t.Fatal(err)
	}

	assertLayerEqual(t, layer3b, layer3)

	// Create again with same name, should return error
	if _, err := ls2.CreateRWLayer("some-mount_name", layer3b.ChainID(), nil); err == nil {
		t.Fatal("Expected error creating mount with same name")
	} else if err != ErrMountNameConflict {
		t.Fatal(err)
	}

	m2, err := ls2.GetRWLayer("some-mount_name")
	if err != nil {
		t.Fatal(err)
	}

	if mountPath, err := m2.Mount(""); err != nil {
		t.Fatal(err)
	} else if path != mountPath {
		t.Fatalf("Unexpected path %s, expected %s", mountPath, path)
	}

	if mountPath, err := m2.Mount(""); err != nil {
		t.Fatal(err)
	} else if path != mountPath {
		t.Fatalf("Unexpected path %s, expected %s", mountPath, path)
	}
	if err := m2.Unmount(); err != nil {
		t.Fatal(err)
	}

	b, err := ioutil.ReadFile(filepath.Join(path, "testfile.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if expected := "nothing here"; string(b) != expected {
		t.Fatalf("Unexpected content %q, expected %q", string(b), expected)
	}

	if err := m2.Unmount(); err != nil {
		t.Fatal(err)
	}

	if metadata, err := ls2.ReleaseRWLayer(m2); err != nil {
		t.Fatal(err)
	} else if len(metadata) != 0 {
		t.Fatalf("Unexpectedly deleted layers: %#v", metadata)
	}

	if metadata, err := ls2.ReleaseRWLayer(m2); err != nil {
		t.Fatal(err)
	} else if len(metadata) != 0 {
		t.Fatalf("Unexpectedly deleted layers: %#v", metadata)
	}

	releaseAndCheckDeleted(t, ls2, layer3b, layer3, layer2, layer1)
}

func TestTarStreamStability(t *testing.T) {
	// TODO Windows: Figure out why this is failing
	if runtime.GOOS == "windows" {
		t.Skip("Failing on Windows")
	}
	ls, _, cleanup := newTestStore(t)
	defer cleanup()

	files1 := []FileApplier{
		newTestFile("/etc/hosts", []byte("mydomain 10.0.0.1"), 0644),
		newTestFile("/etc/profile", []byte("PATH=/usr/bin"), 0644),
	}
	addedFile := newTestFile("/etc/shadow", []byte("root:::::::"), 0644)
	files2 := []FileApplier{
		newTestFile("/etc/hosts", []byte("mydomain 10.0.0.2"), 0644),
		newTestFile("/etc/profile", []byte("PATH=/usr/bin"), 0664),
		newTestFile("/root/.bashrc", []byte("PATH=/usr/sbin:/usr/bin"), 0644),
	}

	tar1, err := tarFromFiles(files1...)
	if err != nil {
		t.Fatal(err)
	}

	tar2, err := tarFromFiles(files2...)
	if err != nil {
		t.Fatal(err)
	}

	layer1, err := ls.Register(bytes.NewReader(tar1), "", Platform(runtime.GOOS))
	if err != nil {
		t.Fatal(err)
	}

	// hack layer to add file
	p, err := ls.(*layerStore).driver.Get(layer1.(*referencedCacheLayer).cacheID, "")
	if err != nil {
		t.Fatal(err)
	}

	if err := addedFile.ApplyFile(p); err != nil {
		t.Fatal(err)
	}

	if err := ls.(*layerStore).driver.Put(layer1.(*referencedCacheLayer).cacheID); err != nil {
		t.Fatal(err)
	}

	layer2, err := ls.Register(bytes.NewReader(tar2), layer1.ChainID(), Platform(runtime.GOOS))
	if err != nil {
		t.Fatal(err)
	}

	id1 := layer1.ChainID()
	t.Logf("Layer 1: %s", layer1.ChainID())
	t.Logf("Layer 2: %s", layer2.ChainID())

	if _, err := ls.Release(layer1); err != nil {
		t.Fatal(err)
	}

	assertLayerDiff(t, tar2, layer2)

	layer1b, err := ls.Get(id1)
	if err != nil {
		t.Logf("Content of layer map: %#v", ls.(*layerStore).layerMap)
		t.Fatal(err)
	}

	if _, err := ls.Release(layer2); err != nil {
		t.Fatal(err)
	}

	assertLayerDiff(t, tar1, layer1b)

	if _, err := ls.Release(layer1b); err != nil {
		t.Fatal(err)
	}
}

func assertLayerDiff(t *testing.T, expected []byte, layer Layer) {
	expectedDigest := digest.FromBytes(expected)

	if digest.Digest(layer.DiffID()) != expectedDigest {
		t.Fatalf("Mismatched diff id for %s, got %s, expected %s", layer.ChainID(), layer.DiffID(), expected)
	}

	ts, err := layer.TarStream()
	if err != nil {
		t.Fatal(err)
	}
	defer ts.Close()

	actual, err := ioutil.ReadAll(ts)
	if err != nil {
		t.Fatal(err)
	}

	if len(actual) != len(expected) {
		logByteDiff(t, actual, expected)
		t.Fatalf("Mismatched tar stream size for %s, got %d, expected %d", layer.ChainID(), len(actual), len(expected))
	}

	actualDigest := digest.FromBytes(actual)

	if actualDigest != expectedDigest {
		logByteDiff(t, actual, expected)
		t.Fatalf("Wrong digest of tar stream, got %s, expected %s", actualDigest, expectedDigest)
	}
}

const maxByteLog = 4 * 1024

func logByteDiff(t *testing.T, actual, expected []byte) {
	d1, d2 := byteDiff(actual, expected)
	if len(d1) == 0 && len(d2) == 0 {
		return
	}

	prefix := len(actual) - len(d1)
	if len(d1) > maxByteLog || len(d2) > maxByteLog {
		t.Logf("Byte diff after %d matching bytes", prefix)
	} else {
		t.Logf("Byte diff after %d matching bytes\nActual bytes after prefix:\n%x\nExpected bytes after prefix:\n%x", prefix, d1, d2)
	}
}

// byteDiff returns the differing bytes after the matching prefix
func byteDiff(b1, b2 []byte) ([]byte, []byte) {
	i := 0
	for i < len(b1) && i < len(b2) {
		if b1[i] != b2[i] {
			break
		}
		i++
	}

	return b1[i:], b2[i:]
}

func tarFromFiles(files ...FileApplier) ([]byte, error) {
	td, err := ioutil.TempDir("", "tar-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(td)

	for _, f := range files {
		if err := f.ApplyFile(td); err != nil {
			return nil, err
		}
	}

	r, err := archive.Tar(td, archive.Uncompressed)
	if err != nil {
		return nil, err
	}

	buf := bytes.NewBuffer(nil)
	if _, err := io.Copy(buf, r); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

// assertReferences asserts that all the references are to the same
// image and represent the full set of references to that image.
func assertReferences(t *testing.T, references ...Layer) {
	if len(references) == 0 {
		return
	}
	base := references[0].(*referencedCacheLayer).roLayer
	seenReferences := map[Layer]struct{}{
		references[0]: {},
	}
	for i := 1; i < len(references); i++ {
		other := references[i].(*referencedCacheLayer).roLayer
		if base != other {
			t.Fatalf("Unexpected referenced cache layer %s, expecting %s", other.ChainID(), base.ChainID())
		}
		if _, ok := base.references[references[i]]; !ok {
			t.Fatalf("Reference not part of reference list: %v", references[i])
		}
		if _, ok := seenReferences[references[i]]; ok {
			t.Fatalf("Duplicated reference %v", references[i])
		}
	}
	if rc := len(base.references); rc != len(references) {
		t.Fatalf("Unexpected number of references %d, expecting %d", rc, len(references))
	}
}

func TestRegisterExistingLayer(t *testing.T) {
	ls, _, cleanup := newTestStore(t)
	defer cleanup()

	baseFiles := []FileApplier{
		newTestFile("/etc/profile", []byte("# Base configuration"), 0644),
	}

	layerFiles := []FileApplier{
		newTestFile("/root/.bashrc", []byte("# Root configuration"), 0644),
	}

	li := initWithFiles(baseFiles...)
	layer1, err := createLayer(ls, "", li)
	if err != nil {
		t.Fatal(err)
	}

	tar1, err := tarFromFiles(layerFiles...)
	if err != nil {
		t.Fatal(err)
	}

	layer2a, err := ls.Register(bytes.NewReader(tar1), layer1.ChainID(), Platform(runtime.GOOS))
	if err != nil {
		t.Fatal(err)
	}

	layer2b, err := ls.Register(bytes.NewReader(tar1), layer1.ChainID(), Platform(runtime.GOOS))
	if err != nil {
		t.Fatal(err)
	}

	assertReferences(t, layer2a, layer2b)
}

func TestTarStreamVerification(t *testing.T) {
	// TODO Windows: Figure out why this is failing
	if runtime.GOOS == "windows" {
		t.Skip("Failing on Windows")
	}
	ls, tmpdir, cleanup := newTestStore(t)
	defer cleanup()

	files1 := []FileApplier{
		newTestFile("/foo", []byte("abc"), 0644),
		newTestFile("/bar", []byte("def"), 0644),
	}
	files2 := []FileApplier{
		newTestFile("/foo", []byte("abc"), 0644),
		newTestFile("/bar", []byte("def"), 0600), // different perm
	}

	tar1, err := tarFromFiles(files1...)
	if err != nil {
		t.Fatal(err)
	}

	tar2, err := tarFromFiles(files2...)
	if err != nil {
		t.Fatal(err)
	}

	layer1, err := ls.Register(bytes.NewReader(tar1), "", Platform(runtime.GOOS))
	if err != nil {
		t.Fatal(err)
	}

	layer2, err := ls.Register(bytes.NewReader(tar2), "", Platform(runtime.GOOS))
	if err != nil {
		t.Fatal(err)
	}
	id1 := digest.Digest(layer1.ChainID())
	id2 := digest.Digest(layer2.ChainID())

	// Replace tar data files
	src, err := os.Open(filepath.Join(tmpdir, id1.Algorithm().String(), id1.Hex(), "tar-split.json.gz"))
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()

	dst, err := os.Create(filepath.Join(tmpdir, id2.Algorithm().String(), id2.Hex(), "tar-split.json.gz"))
	if err != nil {
		t.Fatal(err)
	}
	defer dst.Close()

	if _, err := io.Copy(dst, src); err != nil {
		t.Fatal(err)
	}

	src.Sync()
	dst.Sync()

	ts, err := layer2.TarStream()
	if err != nil {
		t.Fatal(err)
	}
	_, err = io.Copy(ioutil.Discard, ts)
	if err == nil {
		t.Fatal("expected data verification to fail")
	}
	if !strings.Contains(err.Error(), "could not verify layer data") {
		t.Fatalf("wrong error returned from tarstream: %q", err)
	}
}
