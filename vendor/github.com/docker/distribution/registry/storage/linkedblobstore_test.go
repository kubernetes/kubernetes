package storage

import (
	"fmt"
	"io"
	"reflect"
	"strconv"
	"testing"

	"github.com/docker/distribution"
	"github.com/docker/distribution/context"
	"github.com/opencontainers/go-digest"

	"github.com/docker/distribution/reference"
	"github.com/docker/distribution/testutil"
)

func TestLinkedBlobStoreCreateWithMountFrom(t *testing.T) {
	fooRepoName, _ := reference.WithName("nm/foo")
	fooEnv := newManifestStoreTestEnv(t, fooRepoName, "thetag")
	ctx := context.Background()
	stats, err := mockRegistry(t, fooEnv.registry)
	if err != nil {
		t.Fatal(err)
	}

	// Build up some test layers and add them to the manifest, saving the
	// readseekers for upload later.
	testLayers := map[digest.Digest]io.ReadSeeker{}
	for i := 0; i < 2; i++ {
		rs, ds, err := testutil.CreateRandomTarFile()
		if err != nil {
			t.Fatalf("unexpected error generating test layer file")
		}
		dgst := digest.Digest(ds)

		testLayers[digest.Digest(dgst)] = rs
	}

	// upload the layers to foo/bar
	for dgst, rs := range testLayers {
		wr, err := fooEnv.repository.Blobs(fooEnv.ctx).Create(fooEnv.ctx)
		if err != nil {
			t.Fatalf("unexpected error creating test upload: %v", err)
		}

		if _, err := io.Copy(wr, rs); err != nil {
			t.Fatalf("unexpected error copying to upload: %v", err)
		}

		if _, err := wr.Commit(fooEnv.ctx, distribution.Descriptor{Digest: dgst}); err != nil {
			t.Fatalf("unexpected error finishing upload: %v", err)
		}
	}

	// create another repository nm/bar
	barRepoName, _ := reference.WithName("nm/bar")
	barRepo, err := fooEnv.registry.Repository(ctx, barRepoName)
	if err != nil {
		t.Fatalf("unexpected error getting repo: %v", err)
	}

	// cross-repo mount the test layers into a nm/bar
	for dgst := range testLayers {
		fooCanonical, _ := reference.WithDigest(fooRepoName, dgst)
		option := WithMountFrom(fooCanonical)
		// ensure we can instrospect it
		createOpts := distribution.CreateOptions{}
		if err := option.Apply(&createOpts); err != nil {
			t.Fatalf("failed to apply MountFrom option: %v", err)
		}
		if !createOpts.Mount.ShouldMount || createOpts.Mount.From.String() != fooCanonical.String() {
			t.Fatalf("unexpected create options: %#+v", createOpts.Mount)
		}

		_, err := barRepo.Blobs(ctx).Create(ctx, WithMountFrom(fooCanonical))
		if err == nil {
			t.Fatalf("unexpected non-error while mounting from %q: %v", fooRepoName.String(), err)
		}
		if _, ok := err.(distribution.ErrBlobMounted); !ok {
			t.Fatalf("expected ErrMountFrom error, not %T: %v", err, err)
		}
	}
	for dgst := range testLayers {
		fooCanonical, _ := reference.WithDigest(fooRepoName, dgst)
		count, exists := stats[fooCanonical.String()]
		if !exists {
			t.Errorf("expected entry %q not found among handled stat calls", fooCanonical.String())
		} else if count != 1 {
			t.Errorf("expected exactly one stat call for entry %q, not %d", fooCanonical.String(), count)
		}
	}

	clearStats(stats)

	// create yet another repository nm/baz
	bazRepoName, _ := reference.WithName("nm/baz")
	bazRepo, err := fooEnv.registry.Repository(ctx, bazRepoName)
	if err != nil {
		t.Fatalf("unexpected error getting repo: %v", err)
	}

	// cross-repo mount them into a nm/baz and provide a prepopulated blob descriptor
	for dgst := range testLayers {
		fooCanonical, _ := reference.WithDigest(fooRepoName, dgst)
		size, err := strconv.ParseInt("0x"+dgst.Hex()[:8], 0, 64)
		if err != nil {
			t.Fatal(err)
		}
		prepolutatedDescriptor := distribution.Descriptor{
			Digest:    dgst,
			Size:      size,
			MediaType: "application/octet-stream",
		}
		_, err = bazRepo.Blobs(ctx).Create(ctx, WithMountFrom(fooCanonical), &statCrossMountCreateOption{
			desc: prepolutatedDescriptor,
		})
		blobMounted, ok := err.(distribution.ErrBlobMounted)
		if !ok {
			t.Errorf("expected ErrMountFrom error, not %T: %v", err, err)
			continue
		}
		if !reflect.DeepEqual(blobMounted.Descriptor, prepolutatedDescriptor) {
			t.Errorf("unexpected descriptor: %#+v != %#+v", blobMounted.Descriptor, prepolutatedDescriptor)
		}
	}
	// this time no stat calls will be made
	if len(stats) != 0 {
		t.Errorf("unexpected number of stats made: %d != %d", len(stats), len(testLayers))
	}
}

func clearStats(stats map[string]int) {
	for k := range stats {
		delete(stats, k)
	}
}

// mockRegistry sets a mock blob descriptor service factory that overrides
// statter's Stat method to note each attempt to stat a blob in any repository.
// Returned stats map contains canonical references to blobs with a number of
// attempts.
func mockRegistry(t *testing.T, nm distribution.Namespace) (map[string]int, error) {
	registry, ok := nm.(*registry)
	if !ok {
		return nil, fmt.Errorf("not an expected type of registry: %T", nm)
	}
	stats := make(map[string]int)

	registry.blobDescriptorServiceFactory = &mockBlobDescriptorServiceFactory{
		t:     t,
		stats: stats,
	}

	return stats, nil
}

type mockBlobDescriptorServiceFactory struct {
	t     *testing.T
	stats map[string]int
}

func (f *mockBlobDescriptorServiceFactory) BlobAccessController(svc distribution.BlobDescriptorService) distribution.BlobDescriptorService {
	return &mockBlobDescriptorService{
		BlobDescriptorService: svc,
		t:     f.t,
		stats: f.stats,
	}
}

type mockBlobDescriptorService struct {
	distribution.BlobDescriptorService
	t     *testing.T
	stats map[string]int
}

var _ distribution.BlobDescriptorService = &mockBlobDescriptorService{}

func (bs *mockBlobDescriptorService) Stat(ctx context.Context, dgst digest.Digest) (distribution.Descriptor, error) {
	statter, ok := bs.BlobDescriptorService.(*linkedBlobStatter)
	if !ok {
		return distribution.Descriptor{}, fmt.Errorf("unexpected blob descriptor service: %T", bs.BlobDescriptorService)
	}

	name := statter.repository.Named()
	canonical, err := reference.WithDigest(name, dgst)
	if err != nil {
		return distribution.Descriptor{}, fmt.Errorf("failed to make canonical reference: %v", err)
	}

	bs.stats[canonical.String()]++
	bs.t.Logf("calling Stat on %s", canonical.String())

	return bs.BlobDescriptorService.Stat(ctx, dgst)
}

// statCrossMountCreateOptions ensures the expected options type is passed, and optionally pre-fills the cross-mount stat info
type statCrossMountCreateOption struct {
	desc distribution.Descriptor
}

var _ distribution.BlobCreateOption = statCrossMountCreateOption{}

func (f statCrossMountCreateOption) Apply(v interface{}) error {
	opts, ok := v.(*distribution.CreateOptions)
	if !ok {
		return fmt.Errorf("Unexpected create options: %#v", v)
	}

	if !opts.Mount.ShouldMount {
		return nil
	}

	opts.Mount.Stat = &f.desc

	return nil
}
