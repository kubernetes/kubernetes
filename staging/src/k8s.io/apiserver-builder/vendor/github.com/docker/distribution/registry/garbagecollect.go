package registry

import (
	"fmt"
	"os"

	"github.com/docker/distribution"
	"github.com/docker/distribution/context"
	"github.com/docker/distribution/digest"
	"github.com/docker/distribution/manifest/schema1"
	"github.com/docker/distribution/manifest/schema2"
	"github.com/docker/distribution/reference"
	"github.com/docker/distribution/registry/storage"
	"github.com/docker/distribution/registry/storage/driver"
	"github.com/docker/distribution/registry/storage/driver/factory"
	"github.com/docker/libtrust"
	"github.com/spf13/cobra"
)

func emit(format string, a ...interface{}) {
	if dryRun {
		fmt.Printf(format+"\n", a...)
	}
}

func markAndSweep(ctx context.Context, storageDriver driver.StorageDriver, registry distribution.Namespace) error {

	repositoryEnumerator, ok := registry.(distribution.RepositoryEnumerator)
	if !ok {
		return fmt.Errorf("unable to convert Namespace to RepositoryEnumerator")
	}

	// mark
	markSet := make(map[digest.Digest]struct{})
	err := repositoryEnumerator.Enumerate(ctx, func(repoName string) error {
		emit(repoName)

		var err error
		named, err := reference.ParseNamed(repoName)
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

			switch manifest.(type) {
			case *schema1.SignedManifest:
				signaturesGetter, ok := manifestService.(distribution.SignaturesGetter)
				if !ok {
					return fmt.Errorf("unable to convert ManifestService into SignaturesGetter")
				}
				signatures, err := signaturesGetter.GetSignatures(ctx, dgst)
				if err != nil {
					return fmt.Errorf("failed to get signatures for signed manifest: %v", err)
				}
				for _, signatureDigest := range signatures {
					emit("%s: marking signature %s", repoName, signatureDigest)
					markSet[signatureDigest] = struct{}{}
				}
				break
			case *schema2.DeserializedManifest:
				config := manifest.(*schema2.DeserializedManifest).Config
				emit("%s: marking configuration %s", repoName, config.Digest)
				markSet[config.Digest] = struct{}{}
				break
			}

			return nil
		})

		return err
	})

	if err != nil {
		return fmt.Errorf("failed to mark: %v\n", err)
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
	vacuum := storage.NewVacuum(ctx, storageDriver)
	for dgst := range deleteSet {
		emit("blob eligible for deletion: %s", dgst)
		if dryRun {
			continue
		}
		err = vacuum.RemoveBlob(string(dgst))
		if err != nil {
			return fmt.Errorf("failed to delete blob %s: %v\n", dgst, err)
		}
	}

	return err
}

func init() {
	GCCmd.Flags().BoolVarP(&dryRun, "dry-run", "d", false, "do everything expect remove the blobs")
}

var dryRun bool

// GCCmd is the cobra command that corresponds to the garbage-collect subcommand
var GCCmd = &cobra.Command{
	Use:   "garbage-collect <config>",
	Short: "`garbage-collect` deletes layers not referenced by any manifests",
	Long:  "`garbage-collect` deletes layers not referenced by any manifests",
	Run: func(cmd *cobra.Command, args []string) {
		config, err := resolveConfiguration(args)
		if err != nil {
			fmt.Fprintf(os.Stderr, "configuration error: %v\n", err)
			cmd.Usage()
			os.Exit(1)
		}

		driver, err := factory.Create(config.Storage.Type(), config.Storage.Parameters())
		if err != nil {
			fmt.Fprintf(os.Stderr, "failed to construct %s driver: %v", config.Storage.Type(), err)
			os.Exit(1)
		}

		ctx := context.Background()
		ctx, err = configureLogging(ctx, config)
		if err != nil {
			fmt.Fprintf(os.Stderr, "unable to configure logging with config: %s", err)
			os.Exit(1)
		}

		k, err := libtrust.GenerateECP256PrivateKey()
		if err != nil {
			fmt.Fprint(os.Stderr, err)
			os.Exit(1)
		}

		registry, err := storage.NewRegistry(ctx, driver, storage.DisableSchema1Signatures, storage.Schema1SigningKey(k))
		if err != nil {
			fmt.Fprintf(os.Stderr, "failed to construct registry: %v", err)
			os.Exit(1)
		}

		err = markAndSweep(ctx, driver, registry)
		if err != nil {
			fmt.Fprintf(os.Stderr, "failed to garbage collect: %v", err)
			os.Exit(1)
		}
	},
}
