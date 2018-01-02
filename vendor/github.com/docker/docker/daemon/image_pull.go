package daemon

import (
	"io"
	"runtime"
	"strings"

	dist "github.com/docker/distribution"
	"github.com/docker/distribution/reference"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/distribution"
	progressutils "github.com/docker/docker/distribution/utils"
	"github.com/docker/docker/pkg/progress"
	"github.com/docker/docker/registry"
	"github.com/opencontainers/go-digest"
	"golang.org/x/net/context"
)

// PullImage initiates a pull operation. image is the repository name to pull, and
// tag may be either empty, or indicate a specific tag to pull.
func (daemon *Daemon) PullImage(ctx context.Context, image, tag, platform string, metaHeaders map[string][]string, authConfig *types.AuthConfig, outStream io.Writer) error {
	// Special case: "pull -a" may send an image name with a
	// trailing :. This is ugly, but let's not break API
	// compatibility.
	image = strings.TrimSuffix(image, ":")

	ref, err := reference.ParseNormalizedNamed(image)
	if err != nil {
		return err
	}

	if tag != "" {
		// The "tag" could actually be a digest.
		var dgst digest.Digest
		dgst, err = digest.Parse(tag)
		if err == nil {
			ref, err = reference.WithDigest(reference.TrimNamed(ref), dgst)
		} else {
			ref, err = reference.WithTag(ref, tag)
		}
		if err != nil {
			return err
		}
	}

	return daemon.pullImageWithReference(ctx, ref, platform, metaHeaders, authConfig, outStream)
}

func (daemon *Daemon) pullImageWithReference(ctx context.Context, ref reference.Named, platform string, metaHeaders map[string][]string, authConfig *types.AuthConfig, outStream io.Writer) error {
	// Include a buffer so that slow client connections don't affect
	// transfer performance.
	progressChan := make(chan progress.Progress, 100)

	writesDone := make(chan struct{})

	ctx, cancelFunc := context.WithCancel(ctx)

	go func() {
		progressutils.WriteDistributionProgress(cancelFunc, outStream, progressChan)
		close(writesDone)
	}()

	// Default to the host OS platform in case it hasn't been populated with an explicit value.
	if platform == "" {
		platform = runtime.GOOS
	}

	imagePullConfig := &distribution.ImagePullConfig{
		Config: distribution.Config{
			MetaHeaders:      metaHeaders,
			AuthConfig:       authConfig,
			ProgressOutput:   progress.ChanOutput(progressChan),
			RegistryService:  daemon.RegistryService,
			ImageEventLogger: daemon.LogImageEvent,
			MetadataStore:    daemon.stores[platform].distributionMetadataStore,
			ImageStore:       distribution.NewImageConfigStoreFromStore(daemon.stores[platform].imageStore),
			ReferenceStore:   daemon.stores[platform].referenceStore,
		},
		DownloadManager: daemon.downloadManager,
		Schema2Types:    distribution.ImageTypes,
		Platform:        platform,
	}

	err := distribution.Pull(ctx, ref, imagePullConfig)
	close(progressChan)
	<-writesDone
	return err
}

// GetRepository returns a repository from the registry.
func (daemon *Daemon) GetRepository(ctx context.Context, ref reference.Named, authConfig *types.AuthConfig) (dist.Repository, bool, error) {
	// get repository info
	repoInfo, err := daemon.RegistryService.ResolveRepository(ref)
	if err != nil {
		return nil, false, err
	}
	// makes sure name is not empty or `scratch`
	if err := distribution.ValidateRepoName(repoInfo.Name); err != nil {
		return nil, false, err
	}

	// get endpoints
	endpoints, err := daemon.RegistryService.LookupPullEndpoints(reference.Domain(repoInfo.Name))
	if err != nil {
		return nil, false, err
	}

	// retrieve repository
	var (
		confirmedV2 bool
		repository  dist.Repository
		lastError   error
	)

	for _, endpoint := range endpoints {
		if endpoint.Version == registry.APIVersion1 {
			continue
		}

		repository, confirmedV2, lastError = distribution.NewV2Repository(ctx, repoInfo, endpoint, nil, authConfig, "pull")
		if lastError == nil && confirmedV2 {
			break
		}
	}
	return repository, confirmedV2, lastError
}
