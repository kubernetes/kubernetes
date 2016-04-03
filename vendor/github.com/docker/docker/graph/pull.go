package graph

import (
	"fmt"
	"io"

	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/cliconfig"
	"github.com/docker/docker/pkg/streamformatter"
	"github.com/docker/docker/registry"
	"github.com/docker/docker/utils"
)

type ImagePullConfig struct {
	MetaHeaders map[string][]string
	AuthConfig  *cliconfig.AuthConfig
	OutStream   io.Writer
}

type Puller interface {
	// Pull tries to pull the image referenced by `tag`
	// Pull returns an error if any, as well as a boolean that determines whether to retry Pull on the next configured endpoint.
	//
	// TODO(tiborvass): have Pull() take a reference to repository + tag, so that the puller itself is repository-agnostic.
	Pull(tag string) (fallback bool, err error)
}

func NewPuller(s *TagStore, endpoint registry.APIEndpoint, repoInfo *registry.RepositoryInfo, imagePullConfig *ImagePullConfig, sf *streamformatter.StreamFormatter) (Puller, error) {
	switch endpoint.Version {
	case registry.APIVersion2:
		return &v2Puller{
			TagStore: s,
			endpoint: endpoint,
			config:   imagePullConfig,
			sf:       sf,
			repoInfo: repoInfo,
		}, nil
	case registry.APIVersion1:
		return &v1Puller{
			TagStore: s,
			endpoint: endpoint,
			config:   imagePullConfig,
			sf:       sf,
			repoInfo: repoInfo,
		}, nil
	}
	return nil, fmt.Errorf("unknown version %d for registry %s", endpoint.Version, endpoint.URL)
}

func (s *TagStore) Pull(image string, tag string, imagePullConfig *ImagePullConfig) error {
	var sf = streamformatter.NewJSONStreamFormatter()

	// Resolve the Repository name from fqn to RepositoryInfo
	repoInfo, err := s.registryService.ResolveRepository(image)
	if err != nil {
		return err
	}

	// makes sure name is not empty or `scratch`
	if err := validateRepoName(repoInfo.LocalName); err != nil {
		return err
	}

	endpoints, err := s.registryService.LookupEndpoints(repoInfo.CanonicalName)
	if err != nil {
		return err
	}

	logName := repoInfo.LocalName
	if tag != "" {
		logName = utils.ImageReference(logName, tag)
	}

	var (
		lastErr error

		// discardNoSupportErrors is used to track whether an endpoint encountered an error of type registry.ErrNoSupport
		// By default it is false, which means that if a ErrNoSupport error is encountered, it will be saved in lastErr.
		// As soon as another kind of error is encountered, discardNoSupportErrors is set to true, avoiding the saving of
		// any subsequent ErrNoSupport errors in lastErr.
		// It's needed for pull-by-digest on v1 endpoints: if there are only v1 endpoints configured, the error should be
		// returned and displayed, but if there was a v2 endpoint which supports pull-by-digest, then the last relevant
		// error is the ones from v2 endpoints not v1.
		discardNoSupportErrors bool
	)
	for _, endpoint := range endpoints {
		logrus.Debugf("Trying to pull %s from %s %s", repoInfo.LocalName, endpoint.URL, endpoint.Version)

		if !endpoint.Mirror && (endpoint.Official || endpoint.Version == registry.APIVersion2) {
			if repoInfo.Official {
				s.trustService.UpdateBase()
			}
		}

		puller, err := NewPuller(s, endpoint, repoInfo, imagePullConfig, sf)
		if err != nil {
			lastErr = err
			continue
		}
		if fallback, err := puller.Pull(tag); err != nil {
			if fallback {
				if _, ok := err.(registry.ErrNoSupport); !ok {
					// Because we found an error that's not ErrNoSupport, discard all subsequent ErrNoSupport errors.
					discardNoSupportErrors = true
					// save the current error
					lastErr = err
				} else if !discardNoSupportErrors {
					// Save the ErrNoSupport error, because it's either the first error or all encountered errors
					// were also ErrNoSupport errors.
					lastErr = err
				}
				continue
			}
			logrus.Debugf("Not continuing with error: %v", err)
			return err

		}

		s.eventsService.Log("pull", logName, "")
		return nil
	}

	if lastErr == nil {
		lastErr = fmt.Errorf("no endpoints found for %s", image)
	}
	return lastErr
}

func WriteStatus(requestedTag string, out io.Writer, sf *streamformatter.StreamFormatter, layersDownloaded bool) {
	if layersDownloaded {
		out.Write(sf.FormatStatus("", "Status: Downloaded newer image for %s", requestedTag))
	} else {
		out.Write(sf.FormatStatus("", "Status: Image is up to date for %s", requestedTag))
	}
}
