package storage

import (
	"fmt"

	"github.com/docker/distribution"
	"github.com/docker/distribution/context"
	"github.com/docker/distribution/reference"
	"github.com/docker/distribution/registry/storage/driver"
	"github.com/opencontainers/go-digest"
)

func emit(format string, a ...interface{}) {
	fmt.Printf(format+"\n", a...)
}

// MarkAndSweep performs a mark and sweep of registry data
func MarkAndSweep(ctx context.Context, storageDriver driver.StorageDriver, registry distribution.Namespace, dryRun bool) error {
	repositoryEnumerator, ok := registry.(distribution.RepositoryEnumerator)
	if !ok {
		return fmt.Errorf("unable to convert Namespace to RepositoryEnumerator")
	}

	// mark
	markSet := make(map[digest.Digest]struct{})
	err := repositoryEnumerator.Enumerate(ctx, func(repoName string) error {
		emit(repoName)

		var err error
		named, err := reference.WithName(repoName)
		if err != nil {
			return fmt.Errorf("failed to parse repo name %s: %v", repoName, err)
		}
		repository, err := registry.Repository(ctx, named)
		if err != nil {
			return fmt.Errorf("failed to construct repository: %v", err)
		}

		manifestService, err := repository.Manifests(ctx)
		if err != nil {
			return fmt.Errorf("failed to construct manifest service: %v", err)
		}

		manifestEnumerator, ok := manifestService.(distribution.ManifestEnumerator)
		if !ok {
			return fmt.Errorf("unable to convert ManifestService into ManifestEnumerator")
		}

		err = manifestEnumerator.Enumerate(ctx, func(dgst digest.Digest) error {
			// Mark the manifest's blob
			emit("%s: marking manifest %s ", repoName, dgst)
			markSet[dgst] = struct{}{}

			manifest, err := manifestService.Get(ctx, dgst)
			if err != nil {
				return fmt.Errorf("failed to retrieve manifest for digest %v: %v", dgst, err)
			}

			descriptors := manifest.References()
			for _, descriptor := range descriptors {
				markSet[descriptor.Digest] = struct{}{}
				emit("%s: marking blob %s", repoName, descriptor.Digest)
			}

			return nil
		})

		if err != nil {
			// In certain situations such as unfinished uploads, deleting all
			// tags in S3 or removing the _manifests folder manually, this
			// error may be of type PathNotFound.
			//
			// In these cases we can continue marking other manifests safely.
			if _, ok := err.(driver.PathNotFoundError); ok {
				return nil
			}
		}

		return err
	})

	if err != nil {
		return fmt.Errorf("failed to mark: %v", err)
	}

	// sweep
	blobService := registry.Blobs()
	deleteSet := make(map[digest.Digest]struct{})
	err = blobService.Enumerate(ctx, func(dgst digest.Digest) error {
		// check if digest is in markSet. If not, delete it!
		if _, ok := markSet[dgst]; !ok {
			deleteSet[dgst] = struct{}{}
		}
		return nil
	})
	if err != nil {
		return fmt.Errorf("error enumerating blobs: %v", err)
	}
	emit("\n%d blobs marked, %d blobs eligible for deletion", len(markSet), len(deleteSet))
	// Construct vacuum
	vacuum := NewVacuum(ctx, storageDriver)
	for dgst := range deleteSet {
		emit("blob eligible for deletion: %s", dgst)
		if dryRun {
			continue
		}
		err = vacuum.RemoveBlob(string(dgst))
		if err != nil {
			return fmt.Errorf("failed to delete blob %s: %v", dgst, err)
		}
	}

	return err
}
