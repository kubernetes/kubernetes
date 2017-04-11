package proxy

import (
	"io/ioutil"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/docker/distribution"
	"github.com/docker/distribution/context"
	"github.com/docker/distribution/digest"
	"github.com/docker/distribution/reference"
	"github.com/docker/distribution/registry/proxy/scheduler"
	"github.com/docker/distribution/registry/storage"
	"github.com/docker/distribution/registry/storage/cache/memory"
	"github.com/docker/distribution/registry/storage/driver/filesystem"
	"github.com/docker/distribution/registry/storage/driver/inmemory"
)

var sbsMu sync.Mutex

type statsBlobStore struct {
	stats map[string]int
	blobs distribution.BlobStore
}

func (sbs statsBlobStore) Put(ctx context.Context, mediaType string, p []byte) (distribution.Descriptor, error) {
	sbsMu.Lock()
	sbs.stats["put"]++
	sbsMu.Unlock()

	return sbs.blobs.Put(ctx, mediaType, p)
}

func (sbs statsBlobStore) Get(ctx context.Context, dgst digest.Digest) ([]byte, error) {
	sbsMu.Lock()
	sbs.stats["get"]++
	sbsMu.Unlock()

	return sbs.blobs.Get(ctx, dgst)
}

func (sbs statsBlobStore) Create(ctx context.Context, options ...distribution.BlobCreateOption) (distribution.BlobWriter, error) {
	sbsMu.Lock()
	sbs.stats["create"]++
	sbsMu.Unlock()

	return sbs.blobs.Create(ctx, options...)
}

func (sbs statsBlobStore) Resume(ctx context.Context, id string) (distribution.BlobWriter, error) {
	sbsMu.Lock()
	sbs.stats["resume"]++
	sbsMu.Unlock()

	return sbs.blobs.Resume(ctx, id)
}

func (sbs statsBlobStore) Open(ctx context.Context, dgst digest.Digest) (distribution.ReadSeekCloser, error) {
	sbsMu.Lock()
	sbs.stats["open"]++
	sbsMu.Unlock()

	return sbs.blobs.Open(ctx, dgst)
}

func (sbs statsBlobStore) ServeBlob(ctx context.Context, w http.ResponseWriter, r *http.Request, dgst digest.Digest) error {
	sbsMu.Lock()
	sbs.stats["serveblob"]++
	sbsMu.Unlock()

	return sbs.blobs.ServeBlob(ctx, w, r, dgst)
}

func (sbs statsBlobStore) Stat(ctx context.Context, dgst digest.Digest) (distribution.Descriptor, error) {

	sbsMu.Lock()
	sbs.stats["stat"]++
	sbsMu.Unlock()

	return sbs.blobs.Stat(ctx, dgst)
}

func (sbs statsBlobStore) Delete(ctx context.Context, dgst digest.Digest) error {
	sbsMu.Lock()
	sbs.stats["delete"]++
	sbsMu.Unlock()

	return sbs.blobs.Delete(ctx, dgst)
}

type testEnv struct {
	numUnique int
	inRemote  []distribution.Descriptor
	store     proxyBlobStore
	ctx       context.Context
}

func (te *testEnv) LocalStats() *map[string]int {
	sbsMu.Lock()
	ls := te.store.localStore.(statsBlobStore).stats
	sbsMu.Unlock()
	return &ls
}

func (te *testEnv) RemoteStats() *map[string]int {
	sbsMu.Lock()
	rs := te.store.remoteStore.(statsBlobStore).stats
	sbsMu.Unlock()
	return &rs
}

// Populate remote store and record the digests
func makeTestEnv(t *testing.T, name string) *testEnv {
	nameRef, err := reference.ParseNamed(name)
	if err != nil {
		t.Fatalf("unable to parse reference: %s", err)
	}

	ctx := context.Background()

	truthDir, err := ioutil.TempDir("", "truth")
	if err != nil {
		t.Fatalf("unable to create tempdir: %s", err)
	}

	cacheDir, err := ioutil.TempDir("", "cache")
	if err != nil {
		t.Fatalf("unable to create tempdir: %s", err)
	}

	// todo: create a tempfile area here
	localRegistry, err := storage.NewRegistry(ctx, filesystem.New(truthDir), storage.BlobDescriptorCacheProvider(memory.NewInMemoryBlobDescriptorCacheProvider()), storage.EnableRedirect, storage.DisableDigestResumption)
	if err != nil {
		t.Fatalf("error creating registry: %v", err)
	}
	localRepo, err := localRegistry.Repository(ctx, nameRef)
	if err != nil {
		t.Fatalf("unexpected error getting repo: %v", err)
	}

	truthRegistry, err := storage.NewRegistry(ctx, filesystem.New(cacheDir), storage.BlobDescriptorCacheProvider(memory.NewInMemoryBlobDescriptorCacheProvider()))
	if err != nil {
		t.Fatalf("error creating registry: %v", err)
	}
	truthRepo, err := truthRegistry.Repository(ctx, nameRef)
	if err != nil {
		t.Fatalf("unexpected error getting repo: %v", err)
	}

	truthBlobs := statsBlobStore{
		stats: make(map[string]int),
		blobs: truthRepo.Blobs(ctx),
	}

	localBlobs := statsBlobStore{
		stats: make(map[string]int),
		blobs: localRepo.Blobs(ctx),
	}

	s := scheduler.New(ctx, inmemory.New(), "/scheduler-state.json")

	proxyBlobStore := proxyBlobStore{
		repositoryName: nameRef,
		remoteStore:    truthBlobs,
		localStore:     localBlobs,
		scheduler:      s,
		authChallenger: &mockChallenger{},
	}

	te := &testEnv{
		store: proxyBlobStore,
		ctx:   ctx,
	}
	return te
}

func makeBlob(size int) []byte {
	blob := make([]byte, size, size)
	for i := 0; i < size; i++ {
		blob[i] = byte('A' + rand.Int()%48)
	}
	return blob
}

func init() {
	rand.Seed(42)
}

func perm(m []distribution.Descriptor) []distribution.Descriptor {
	for i := 0; i < len(m); i++ {
		j := rand.Intn(i + 1)
		tmp := m[i]
		m[i] = m[j]
		m[j] = tmp
	}
	return m
}

func populate(t *testing.T, te *testEnv, blobCount, size, numUnique int) {
	var inRemote []distribution.Descriptor

	for i := 0; i < numUnique; i++ {
		bytes := makeBlob(size)
		for j := 0; j < blobCount/numUnique; j++ {
			desc, err := te.store.remoteStore.Put(te.ctx, "", bytes)
			if err != nil {
				t.Fatalf("Put in store")
			}

			inRemote = append(inRemote, desc)
		}
	}

	te.inRemote = inRemote
	te.numUnique = numUnique
}
func TestProxyStoreGet(t *testing.T) {
	te := makeTestEnv(t, "foo/bar")

	localStats := te.LocalStats()
	remoteStats := te.RemoteStats()

	populate(t, te, 1, 10, 1)
	_, err := te.store.Get(te.ctx, te.inRemote[0].Digest)
	if err != nil {
		t.Fatal(err)
	}

	if (*localStats)["get"] != 1 && (*localStats)["put"] != 1 {
		t.Errorf("Unexpected local counts")
	}

	if (*remoteStats)["get"] != 1 {
		t.Errorf("Unexpected remote get count")
	}

	_, err = te.store.Get(te.ctx, te.inRemote[0].Digest)
	if err != nil {
		t.Fatal(err)
	}

	if (*localStats)["get"] != 2 && (*localStats)["put"] != 1 {
		t.Errorf("Unexpected local counts")
	}

	if (*remoteStats)["get"] != 1 {
		t.Errorf("Unexpected remote get count")
	}

}

func TestProxyStoreStat(t *testing.T) {
	te := makeTestEnv(t, "foo/bar")

	remoteBlobCount := 1
	populate(t, te, remoteBlobCount, 10, 1)

	localStats := te.LocalStats()
	remoteStats := te.RemoteStats()

	// Stat - touches both stores
	for _, d := range te.inRemote {
		_, err := te.store.Stat(te.ctx, d.Digest)
		if err != nil {
			t.Fatalf("Error stating proxy store")
		}
	}

	if (*localStats)["stat"] != remoteBlobCount {
		t.Errorf("Unexpected local stat count")
	}

	if (*remoteStats)["stat"] != remoteBlobCount {
		t.Errorf("Unexpected remote stat count")
	}

	if te.store.authChallenger.(*mockChallenger).count != len(te.inRemote) {
		t.Fatalf("Unexpected auth challenge count, got %#v", te.store.authChallenger)
	}

}

func TestProxyStoreServeHighConcurrency(t *testing.T) {
	te := makeTestEnv(t, "foo/bar")
	blobSize := 200
	blobCount := 10
	numUnique := 1
	populate(t, te, blobCount, blobSize, numUnique)

	numClients := 16
	testProxyStoreServe(t, te, numClients)
}

func TestProxyStoreServeMany(t *testing.T) {
	te := makeTestEnv(t, "foo/bar")
	blobSize := 200
	blobCount := 10
	numUnique := 4
	populate(t, te, blobCount, blobSize, numUnique)

	numClients := 4
	testProxyStoreServe(t, te, numClients)
}

// todo(richardscothern): blobCount must be smaller than num clients
func TestProxyStoreServeBig(t *testing.T) {
	te := makeTestEnv(t, "foo/bar")

	blobSize := 2 << 20
	blobCount := 4
	numUnique := 2
	populate(t, te, blobCount, blobSize, numUnique)

	numClients := 4
	testProxyStoreServe(t, te, numClients)
}

// testProxyStoreServe will create clients to consume all blobs
// populated in the truth store
func testProxyStoreServe(t *testing.T, te *testEnv, numClients int) {
	localStats := te.LocalStats()
	remoteStats := te.RemoteStats()

	var wg sync.WaitGroup

	for i := 0; i < numClients; i++ {
		// Serveblob - pulls through blobs
		wg.Add(1)
		go func() {
			defer wg.Done()
			for _, remoteBlob := range te.inRemote {
				w := httptest.NewRecorder()
				r, err := http.NewRequest("GET", "", nil)
				if err != nil {
					t.Fatal(err)
				}

				err = te.store.ServeBlob(te.ctx, w, r, remoteBlob.Digest)
				if err != nil {
					t.Fatalf(err.Error())
				}

				bodyBytes := w.Body.Bytes()
				localDigest := digest.FromBytes(bodyBytes)
				if localDigest != remoteBlob.Digest {
					t.Fatalf("Mismatching blob fetch from proxy")
				}
			}
		}()
	}

	wg.Wait()

	remoteBlobCount := len(te.inRemote)
	if (*localStats)["stat"] != remoteBlobCount*numClients && (*localStats)["create"] != te.numUnique {
		t.Fatal("Expected: stat:", remoteBlobCount*numClients, "create:", remoteBlobCount)
	}

	// Wait for any async storage goroutines to finish
	time.Sleep(3 * time.Second)

	remoteStatCount := (*remoteStats)["stat"]
	remoteOpenCount := (*remoteStats)["open"]

	// Serveblob - blobs come from local
	for _, dr := range te.inRemote {
		w := httptest.NewRecorder()
		r, err := http.NewRequest("GET", "", nil)
		if err != nil {
			t.Fatal(err)
		}

		err = te.store.ServeBlob(te.ctx, w, r, dr.Digest)
		if err != nil {
			t.Fatalf(err.Error())
		}

		dl := digest.FromBytes(w.Body.Bytes())
		if dl != dr.Digest {
			t.Errorf("Mismatching blob fetch from proxy")
		}
	}

	localStats = te.LocalStats()
	remoteStats = te.RemoteStats()

	// Ensure remote unchanged
	if (*remoteStats)["stat"] != remoteStatCount && (*remoteStats)["open"] != remoteOpenCount {
		t.Fatalf("unexpected remote stats: %#v", remoteStats)
	}
}
