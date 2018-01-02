package storage

import (
	"io"
	"path"
	"testing"

	"github.com/docker/distribution"
	"github.com/docker/distribution/context"
	"github.com/docker/distribution/reference"
	"github.com/docker/distribution/registry/storage/driver"
	"github.com/docker/distribution/registry/storage/driver/inmemory"
	"github.com/docker/distribution/testutil"
	"github.com/docker/libtrust"
	"github.com/opencontainers/go-digest"
)

type image struct {
	manifest       distribution.Manifest
	manifestDigest digest.Digest
	layers         map[digest.Digest]io.ReadSeeker
}

func createRegistry(t *testing.T, driver driver.StorageDriver, options ...RegistryOption) distribution.Namespace {
	ctx := context.Background()
	k, err := libtrust.GenerateECP256PrivateKey()
	if err != nil {
		t.Fatal(err)
	}
	options = append([]RegistryOption{EnableDelete, Schema1SigningKey(k)}, options...)
	registry, err := NewRegistry(ctx, driver, options...)
	if err != nil {
		t.Fatalf("Failed to construct namespace")
	}
	return registry
}

func makeRepository(t *testing.T, registry distribution.Namespace, name string) distribution.Repository {
	ctx := context.Background()

	// Initialize a dummy repository
	named, err := reference.WithName(name)
	if err != nil {
		t.Fatalf("Failed to parse name %s:  %v", name, err)
	}

	repo, err := registry.Repository(ctx, named)
	if err != nil {
		t.Fatalf("Failed to construct repository: %v", err)
	}
	return repo
}

func makeManifestService(t *testing.T, repository distribution.Repository) distribution.ManifestService {
	ctx := context.Background()

	manifestService, err := repository.Manifests(ctx)
	if err != nil {
		t.Fatalf("Failed to construct manifest store: %v", err)
	}
	return manifestService
}

func allBlobs(t *testing.T, registry distribution.Namespace) map[digest.Digest]struct{} {
	ctx := context.Background()
	blobService := registry.Blobs()
	allBlobsMap := make(map[digest.Digest]struct{})
	err := blobService.Enumerate(ctx, func(dgst digest.Digest) error {
		allBlobsMap[dgst] = struct{}{}
		return nil
	})
	if err != nil {
		t.Fatalf("Error getting all blobs: %v", err)
	}
	return allBlobsMap
}

func uploadImage(t *testing.T, repository distribution.Repository, im image) digest.Digest {
	// upload layers
	err := testutil.UploadBlobs(repository, im.layers)
	if err != nil {
		t.Fatalf("layer upload failed: %v", err)
	}

	// upload manifest
	ctx := context.Background()
	manifestService := makeManifestService(t, repository)
	manifestDigest, err := manifestService.Put(ctx, im.manifest)
	if err != nil {
		t.Fatalf("manifest upload failed: %v", err)
	}

	return manifestDigest
}

func uploadRandomSchema1Image(t *testing.T, repository distribution.Repository) image {
	randomLayers, err := testutil.CreateRandomLayers(2)
	if err != nil {
		t.Fatalf("%v", err)
	}

	digests := []digest.Digest{}
	for digest := range randomLayers {
		digests = append(digests, digest)
	}

	manifest, err := testutil.MakeSchema1Manifest(digests)
	if err != nil {
		t.Fatalf("%v", err)
	}

	manifestDigest := uploadImage(t, repository, image{manifest: manifest, layers: randomLayers})
	return image{
		manifest:       manifest,
		manifestDigest: manifestDigest,
		layers:         randomLayers,
	}
}

func uploadRandomSchema2Image(t *testing.T, repository distribution.Repository) image {
	randomLayers, err := testutil.CreateRandomLayers(2)
	if err != nil {
		t.Fatalf("%v", err)
	}

	digests := []digest.Digest{}
	for digest := range randomLayers {
		digests = append(digests, digest)
	}

	manifest, err := testutil.MakeSchema2Manifest(repository, digests)
	if err != nil {
		t.Fatalf("%v", err)
	}

	manifestDigest := uploadImage(t, repository, image{manifest: manifest, layers: randomLayers})
	return image{
		manifest:       manifest,
		manifestDigest: manifestDigest,
		layers:         randomLayers,
	}
}

func TestNoDeletionNoEffect(t *testing.T) {
	ctx := context.Background()
	inmemoryDriver := inmemory.New()

	registry := createRegistry(t, inmemoryDriver)
	repo := makeRepository(t, registry, "palailogos")
	manifestService, err := repo.Manifests(ctx)

	image1 := uploadRandomSchema1Image(t, repo)
	image2 := uploadRandomSchema1Image(t, repo)
	uploadRandomSchema2Image(t, repo)

	// construct manifestlist for fun.
	blobstatter := registry.BlobStatter()
	manifestList, err := testutil.MakeManifestList(blobstatter, []digest.Digest{
		image1.manifestDigest, image2.manifestDigest})
	if err != nil {
		t.Fatalf("Failed to make manifest list: %v", err)
	}

	_, err = manifestService.Put(ctx, manifestList)
	if err != nil {
		t.Fatalf("Failed to add manifest list: %v", err)
	}

	before := allBlobs(t, registry)

	// Run GC
	err = MarkAndSweep(context.Background(), inmemoryDriver, registry, false)
	if err != nil {
		t.Fatalf("Failed mark and sweep: %v", err)
	}

	after := allBlobs(t, registry)
	if len(before) != len(after) {
		t.Fatalf("Garbage collection affected storage: %d != %d", len(before), len(after))
	}
}

func TestGCWithMissingManifests(t *testing.T) {
	ctx := context.Background()
	d := inmemory.New()

	registry := createRegistry(t, d)
	repo := makeRepository(t, registry, "testrepo")
	uploadRandomSchema1Image(t, repo)

	// Simulate a missing _manifests directory
	revPath, err := pathFor(manifestRevisionsPathSpec{"testrepo"})
	if err != nil {
		t.Fatal(err)
	}

	_manifestsPath := path.Dir(revPath)
	err = d.Delete(ctx, _manifestsPath)
	if err != nil {
		t.Fatal(err)
	}

	err = MarkAndSweep(context.Background(), d, registry, false)
	if err != nil {
		t.Fatalf("Failed mark and sweep: %v", err)
	}

	blobs := allBlobs(t, registry)
	if len(blobs) > 0 {
		t.Errorf("unexpected blobs after gc")
	}
}

func TestDeletionHasEffect(t *testing.T) {
	ctx := context.Background()
	inmemoryDriver := inmemory.New()

	registry := createRegistry(t, inmemoryDriver)
	repo := makeRepository(t, registry, "komnenos")
	manifests, err := repo.Manifests(ctx)

	image1 := uploadRandomSchema1Image(t, repo)
	image2 := uploadRandomSchema1Image(t, repo)
	image3 := uploadRandomSchema2Image(t, repo)

	manifests.Delete(ctx, image2.manifestDigest)
	manifests.Delete(ctx, image3.manifestDigest)

	// Run GC
	err = MarkAndSweep(context.Background(), inmemoryDriver, registry, false)
	if err != nil {
		t.Fatalf("Failed mark and sweep: %v", err)
	}

	blobs := allBlobs(t, registry)

	// check that the image1 manifest and all the layers are still in blobs
	if _, ok := blobs[image1.manifestDigest]; !ok {
		t.Fatalf("First manifest is missing")
	}

	for layer := range image1.layers {
		if _, ok := blobs[layer]; !ok {
			t.Fatalf("manifest 1 layer is missing: %v", layer)
		}
	}

	// check that image2 and image3 layers are not still around
	for layer := range image2.layers {
		if _, ok := blobs[layer]; ok {
			t.Fatalf("manifest 2 layer is present: %v", layer)
		}
	}

	for layer := range image3.layers {
		if _, ok := blobs[layer]; ok {
			t.Fatalf("manifest 3 layer is present: %v", layer)
		}
	}
}

func getAnyKey(digests map[digest.Digest]io.ReadSeeker) (d digest.Digest) {
	for d = range digests {
		break
	}
	return
}

func getKeys(digests map[digest.Digest]io.ReadSeeker) (ds []digest.Digest) {
	for d := range digests {
		ds = append(ds, d)
	}
	return
}

func TestDeletionWithSharedLayer(t *testing.T) {
	ctx := context.Background()
	inmemoryDriver := inmemory.New()

	registry := createRegistry(t, inmemoryDriver)
	repo := makeRepository(t, registry, "tzimiskes")

	// Create random layers
	randomLayers1, err := testutil.CreateRandomLayers(3)
	if err != nil {
		t.Fatalf("failed to make layers: %v", err)
	}

	randomLayers2, err := testutil.CreateRandomLayers(3)
	if err != nil {
		t.Fatalf("failed to make layers: %v", err)
	}

	// Upload all layers
	err = testutil.UploadBlobs(repo, randomLayers1)
	if err != nil {
		t.Fatalf("failed to upload layers: %v", err)
	}

	err = testutil.UploadBlobs(repo, randomLayers2)
	if err != nil {
		t.Fatalf("failed to upload layers: %v", err)
	}

	// Construct manifests
	manifest1, err := testutil.MakeSchema1Manifest(getKeys(randomLayers1))
	if err != nil {
		t.Fatalf("failed to make manifest: %v", err)
	}

	sharedKey := getAnyKey(randomLayers1)
	manifest2, err := testutil.MakeSchema2Manifest(repo, append(getKeys(randomLayers2), sharedKey))
	if err != nil {
		t.Fatalf("failed to make manifest: %v", err)
	}

	manifestService := makeManifestService(t, repo)

	// Upload manifests
	_, err = manifestService.Put(ctx, manifest1)
	if err != nil {
		t.Fatalf("manifest upload failed: %v", err)
	}

	manifestDigest2, err := manifestService.Put(ctx, manifest2)
	if err != nil {
		t.Fatalf("manifest upload failed: %v", err)
	}

	// delete
	err = manifestService.Delete(ctx, manifestDigest2)
	if err != nil {
		t.Fatalf("manifest deletion failed: %v", err)
	}

	// check that all of the layers in layer 1 are still there
	blobs := allBlobs(t, registry)
	for dgst := range randomLayers1 {
		if _, ok := blobs[dgst]; !ok {
			t.Fatalf("random layer 1 blob missing: %v", dgst)
		}
	}
}

func TestOrphanBlobDeleted(t *testing.T) {
	inmemoryDriver := inmemory.New()

	registry := createRegistry(t, inmemoryDriver)
	repo := makeRepository(t, registry, "michael_z_doukas")

	digests, err := testutil.CreateRandomLayers(1)
	if err != nil {
		t.Fatalf("Failed to create random digest: %v", err)
	}

	if err = testutil.UploadBlobs(repo, digests); err != nil {
		t.Fatalf("Failed to upload blob: %v", err)
	}

	// formality to create the necessary directories
	uploadRandomSchema2Image(t, repo)

	// Run GC
	err = MarkAndSweep(context.Background(), inmemoryDriver, registry, false)
	if err != nil {
		t.Fatalf("Failed mark and sweep: %v", err)
	}

	blobs := allBlobs(t, registry)

	// check that orphan blob layers are not still around
	for dgst := range digests {
		if _, ok := blobs[dgst]; ok {
			t.Fatalf("Orphan layer is present: %v", dgst)
		}
	}
}
