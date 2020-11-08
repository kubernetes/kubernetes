package client // import "github.com/docker/docker/client"

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/docker/distribution/reference"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/swarm"
	digest "github.com/opencontainers/go-digest"
	"github.com/pkg/errors"
)

// ServiceCreate creates a new Service.
func (cli *Client) ServiceCreate(ctx context.Context, service swarm.ServiceSpec, options types.ServiceCreateOptions) (types.ServiceCreateResponse, error) {
	var distErr error

	headers := map[string][]string{
		"version": {cli.version},
	}

	if options.EncodedRegistryAuth != "" {
		headers["X-Registry-Auth"] = []string{options.EncodedRegistryAuth}
	}

	// Make sure containerSpec is not nil when no runtime is set or the runtime is set to container
	if service.TaskTemplate.ContainerSpec == nil && (service.TaskTemplate.Runtime == "" || service.TaskTemplate.Runtime == swarm.RuntimeContainer) {
		service.TaskTemplate.ContainerSpec = &swarm.ContainerSpec{}
	}

	if err := validateServiceSpec(service); err != nil {
		return types.ServiceCreateResponse{}, err
	}

	// ensure that the image is tagged
	var imgPlatforms []swarm.Platform
	if service.TaskTemplate.ContainerSpec != nil {
		if taggedImg := imageWithTagString(service.TaskTemplate.ContainerSpec.Image); taggedImg != "" {
			service.TaskTemplate.ContainerSpec.Image = taggedImg
		}
		if options.QueryRegistry {
			var img string
			img, imgPlatforms, distErr = imageDigestAndPlatforms(ctx, cli, service.TaskTemplate.ContainerSpec.Image, options.EncodedRegistryAuth)
			if img != "" {
				service.TaskTemplate.ContainerSpec.Image = img
			}
		}
	}

	// ensure that the image is tagged
	if service.TaskTemplate.PluginSpec != nil {
		if taggedImg := imageWithTagString(service.TaskTemplate.PluginSpec.Remote); taggedImg != "" {
			service.TaskTemplate.PluginSpec.Remote = taggedImg
		}
		if options.QueryRegistry {
			var img string
			img, imgPlatforms, distErr = imageDigestAndPlatforms(ctx, cli, service.TaskTemplate.PluginSpec.Remote, options.EncodedRegistryAuth)
			if img != "" {
				service.TaskTemplate.PluginSpec.Remote = img
			}
		}
	}

	if service.TaskTemplate.Placement == nil && len(imgPlatforms) > 0 {
		service.TaskTemplate.Placement = &swarm.Placement{}
	}
	if len(imgPlatforms) > 0 {
		service.TaskTemplate.Placement.Platforms = imgPlatforms
	}

	var response types.ServiceCreateResponse
	resp, err := cli.post(ctx, "/services/create", nil, service, headers)
	defer ensureReaderClosed(resp)
	if err != nil {
		return response, err
	}

	err = json.NewDecoder(resp.body).Decode(&response)

	if distErr != nil {
		response.Warnings = append(response.Warnings, digestWarning(service.TaskTemplate.ContainerSpec.Image))
	}

	return response, err
}

func imageDigestAndPlatforms(ctx context.Context, cli DistributionAPIClient, image, encodedAuth string) (string, []swarm.Platform, error) {
	distributionInspect, err := cli.DistributionInspect(ctx, image, encodedAuth)
	var platforms []swarm.Platform
	if err != nil {
		return "", nil, err
	}

	imageWithDigest := imageWithDigestString(image, distributionInspect.Descriptor.Digest)

	if len(distributionInspect.Platforms) > 0 {
		platforms = make([]swarm.Platform, 0, len(distributionInspect.Platforms))
		for _, p := range distributionInspect.Platforms {
			// clear architecture field for arm. This is a temporary patch to address
			// https://github.com/docker/swarmkit/issues/2294. The issue is that while
			// image manifests report "arm" as the architecture, the node reports
			// something like "armv7l" (includes the variant), which causes arm images
			// to stop working with swarm mode. This patch removes the architecture
			// constraint for arm images to ensure tasks get scheduled.
			arch := p.Architecture
			if strings.ToLower(arch) == "arm" {
				arch = ""
			}
			platforms = append(platforms, swarm.Platform{
				Architecture: arch,
				OS:           p.OS,
			})
		}
	}
	return imageWithDigest, platforms, err
}

// imageWithDigestString takes an image string and a digest, and updates
// the image string if it didn't originally contain a digest. It returns
// an empty string if there are no updates.
func imageWithDigestString(image string, dgst digest.Digest) string {
	namedRef, err := reference.ParseNormalizedNamed(image)
	if err == nil {
		if _, isCanonical := namedRef.(reference.Canonical); !isCanonical {
			// ensure that image gets a default tag if none is provided
			img, err := reference.WithDigest(namedRef, dgst)
			if err == nil {
				return reference.FamiliarString(img)
			}
		}
	}
	return ""
}

// imageWithTagString takes an image string, and returns a tagged image
// string, adding a 'latest' tag if one was not provided. It returns an
// empty string if a canonical reference was provided
func imageWithTagString(image string) string {
	namedRef, err := reference.ParseNormalizedNamed(image)
	if err == nil {
		return reference.FamiliarString(reference.TagNameOnly(namedRef))
	}
	return ""
}

// digestWarning constructs a formatted warning string using the
// image name that could not be pinned by digest. The formatting
// is hardcoded, but could me made smarter in the future
func digestWarning(image string) string {
	return fmt.Sprintf("image %s could not be accessed on a registry to record\nits digest. Each node will access %s independently,\npossibly leading to different nodes running different\nversions of the image.\n", image, image)
}

func validateServiceSpec(s swarm.ServiceSpec) error {
	if s.TaskTemplate.ContainerSpec != nil && s.TaskTemplate.PluginSpec != nil {
		return errors.New("must not specify both a container spec and a plugin spec in the task template")
	}
	if s.TaskTemplate.PluginSpec != nil && s.TaskTemplate.Runtime != swarm.RuntimePlugin {
		return errors.New("mismatched runtime with plugin spec")
	}
	if s.TaskTemplate.ContainerSpec != nil && (s.TaskTemplate.Runtime != "" && s.TaskTemplate.Runtime != swarm.RuntimeContainer) {
		return errors.New("mismatched runtime with container spec")
	}
	return nil
}
