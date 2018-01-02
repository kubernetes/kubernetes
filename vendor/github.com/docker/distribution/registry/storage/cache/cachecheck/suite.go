package cachecheck

import (
	"reflect"
	"testing"

	"github.com/docker/distribution"
	"github.com/docker/distribution/context"
	"github.com/docker/distribution/registry/storage/cache"
	"github.com/opencontainers/go-digest"
)

// CheckBlobDescriptorCache takes a cache implementation through a common set
// of operations. If adding new tests, please add them here so new
// implementations get the benefit. This should be used for unit tests.
func CheckBlobDescriptorCache(t *testing.T, provider cache.BlobDescriptorCacheProvider) {
	ctx := context.Background()

	checkBlobDescriptorCacheEmptyRepository(ctx, t, provider)
	checkBlobDescriptorCacheSetAndRead(ctx, t, provider)
	checkBlobDescriptorCacheClear(ctx, t, provider)
}

func checkBlobDescriptorCacheEmptyRepository(ctx context.Context, t *testing.T, provider cache.BlobDescriptorCacheProvider) {
	if _, err := provider.Stat(ctx, "sha384:abc111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111"); err != distribution.ErrBlobUnknown {
		t.Fatalf("expected unknown blob error with empty store: %v", err)
	}

	cache, err := provider.RepositoryScoped("")
	if err == nil {
		t.Fatalf("expected an error when asking for invalid repo")
	}

	cache, err = provider.RepositoryScoped("foo/bar")
	if err != nil {
		t.Fatalf("unexpected error getting repository: %v", err)
	}

	if err := cache.SetDescriptor(ctx, "", distribution.Descriptor{
		Digest:    "sha384:abc",
		Size:      10,
		MediaType: "application/octet-stream"}); err != digest.ErrDigestInvalidFormat {
		t.Fatalf("expected error with invalid digest: %v", err)
	}

	if err := cache.SetDescriptor(ctx, "sha384:abc111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", distribution.Descriptor{
		Digest:    "",
		Size:      10,
		MediaType: "application/octet-stream"}); err == nil {
		t.Fatalf("expected error setting value on invalid descriptor")
	}

	if _, err := cache.Stat(ctx, ""); err != digest.ErrDigestInvalidFormat {
		t.Fatalf("expected error checking for cache item with empty digest: %v", err)
	}

	if _, err := cache.Stat(ctx, "sha384:abc111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111"); err != distribution.ErrBlobUnknown {
		t.Fatalf("expected unknown blob error with empty repo: %v", err)
	}
}

func checkBlobDescriptorCacheSetAndRead(ctx context.Context, t *testing.T, provider cache.BlobDescriptorCacheProvider) {
	localDigest := digest.Digest("sha384:abc111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111")
	expected := distribution.Descriptor{
		Digest:    "sha256:abc1111111111111111111111111111111111111111111111111111111111111",
		Size:      10,
		MediaType: "application/octet-stream"}

	cache, err := provider.RepositoryScoped("foo/bar")
	if err != nil {
		t.Fatalf("unexpected error getting scoped cache: %v", err)
	}

	if err := cache.SetDescriptor(ctx, localDigest, expected); err != nil {
		t.Fatalf("error setting descriptor: %v", err)
	}

	desc, err := cache.Stat(ctx, localDigest)
	if err != nil {
		t.Fatalf("unexpected error statting fake2:abc: %v", err)
	}

	if !reflect.DeepEqual(expected, desc) {
		t.Fatalf("unexpected descriptor: %#v != %#v", expected, desc)
	}

	// also check that we set the canonical key ("fake:abc")
	desc, err = cache.Stat(ctx, localDigest)
	if err != nil {
		t.Fatalf("descriptor not returned for canonical key: %v", err)
	}

	if !reflect.DeepEqual(expected, desc) {
		t.Fatalf("unexpected descriptor: %#v != %#v", expected, desc)
	}

	// ensure that global gets extra descriptor mapping
	desc, err = provider.Stat(ctx, localDigest)
	if err != nil {
		t.Fatalf("expected blob unknown in global cache: %v, %v", err, desc)
	}

	if !reflect.DeepEqual(desc, expected) {
		t.Fatalf("unexpected descriptor: %#v != %#v", expected, desc)
	}

	// get at it through canonical descriptor
	desc, err = provider.Stat(ctx, expected.Digest)
	if err != nil {
		t.Fatalf("unexpected error checking glboal descriptor: %v", err)
	}

	if !reflect.DeepEqual(desc, expected) {
		t.Fatalf("unexpected descriptor: %#v != %#v", expected, desc)
	}

	// now, we set the repo local mediatype to something else and ensure it
	// doesn't get changed in the provider cache.
	expected.MediaType = "application/json"

	if err := cache.SetDescriptor(ctx, localDigest, expected); err != nil {
		t.Fatalf("unexpected error setting descriptor: %v", err)
	}

	desc, err = cache.Stat(ctx, localDigest)
	if err != nil {
		t.Fatalf("unexpected error getting descriptor: %v", err)
	}

	if !reflect.DeepEqual(desc, expected) {
		t.Fatalf("unexpected descriptor: %#v != %#v", desc, expected)
	}

	desc, err = provider.Stat(ctx, localDigest)
	if err != nil {
		t.Fatalf("unexpected error getting global descriptor: %v", err)
	}

	expected.MediaType = "application/octet-stream" // expect original mediatype in global

	if !reflect.DeepEqual(desc, expected) {
		t.Fatalf("unexpected descriptor: %#v != %#v", desc, expected)
	}
}

func checkBlobDescriptorCacheClear(ctx context.Context, t *testing.T, provider cache.BlobDescriptorCacheProvider) {
	localDigest := digest.Digest("sha384:def111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111")
	expected := distribution.Descriptor{
		Digest:    "sha256:def1111111111111111111111111111111111111111111111111111111111111",
		Size:      10,
		MediaType: "application/octet-stream"}

	cache, err := provider.RepositoryScoped("foo/bar")
	if err != nil {
		t.Fatalf("unexpected error getting scoped cache: %v", err)
	}

	if err := cache.SetDescriptor(ctx, localDigest, expected); err != nil {
		t.Fatalf("error setting descriptor: %v", err)
	}

	desc, err := cache.Stat(ctx, localDigest)
	if err != nil {
		t.Fatalf("unexpected error statting fake2:abc: %v", err)
	}

	if !reflect.DeepEqual(expected, desc) {
		t.Fatalf("unexpected descriptor: %#v != %#v", expected, desc)
	}

	err = cache.Clear(ctx, localDigest)
	if err != nil {
		t.Error(err)
	}

	desc, err = cache.Stat(ctx, localDigest)
	if err == nil {
		t.Fatalf("expected error statting deleted blob: %v", err)
	}
}
