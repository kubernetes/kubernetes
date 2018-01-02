package daemon

import (
	"runtime"
	"time"

	"github.com/docker/distribution/reference"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/layer"
	"github.com/pkg/errors"
)

// LookupImage looks up an image by name and returns it as an ImageInspect
// structure.
func (daemon *Daemon) LookupImage(name string) (*types.ImageInspect, error) {
	img, err := daemon.GetImage(name)
	if err != nil {
		return nil, errors.Wrapf(err, "no such image: %s", name)
	}

	// If the image OS isn't set, assume it's the host OS
	platform := img.OS
	if platform == "" {
		platform = runtime.GOOS
	}

	refs := daemon.stores[platform].referenceStore.References(img.ID().Digest())
	repoTags := []string{}
	repoDigests := []string{}
	for _, ref := range refs {
		switch ref.(type) {
		case reference.NamedTagged:
			repoTags = append(repoTags, reference.FamiliarString(ref))
		case reference.Canonical:
			repoDigests = append(repoDigests, reference.FamiliarString(ref))
		}
	}

	var size int64
	var layerMetadata map[string]string
	layerID := img.RootFS.ChainID()
	if layerID != "" {
		l, err := daemon.stores[platform].layerStore.Get(layerID)
		if err != nil {
			return nil, err
		}
		defer layer.ReleaseAndLog(daemon.stores[platform].layerStore, l)
		size, err = l.Size()
		if err != nil {
			return nil, err
		}

		layerMetadata, err = l.Metadata()
		if err != nil {
			return nil, err
		}
	}

	comment := img.Comment
	if len(comment) == 0 && len(img.History) > 0 {
		comment = img.History[len(img.History)-1].Comment
	}

	lastUpdated, err := daemon.stores[platform].imageStore.GetLastUpdated(img.ID())
	if err != nil {
		return nil, err
	}

	imageInspect := &types.ImageInspect{
		ID:              img.ID().String(),
		RepoTags:        repoTags,
		RepoDigests:     repoDigests,
		Parent:          img.Parent.String(),
		Comment:         comment,
		Created:         img.Created.Format(time.RFC3339Nano),
		Container:       img.Container,
		ContainerConfig: &img.ContainerConfig,
		DockerVersion:   img.DockerVersion,
		Author:          img.Author,
		Config:          img.Config,
		Architecture:    img.Architecture,
		Os:              platform,
		OsVersion:       img.OSVersion,
		Size:            size,
		VirtualSize:     size, // TODO: field unused, deprecate
		RootFS:          rootFSToAPIType(img.RootFS),
		Metadata: types.ImageMetadata{
			LastTagTime: lastUpdated,
		},
	}

	imageInspect.GraphDriver.Name = daemon.GraphDriverName(platform)
	imageInspect.GraphDriver.Data = layerMetadata

	return imageInspect, nil
}
