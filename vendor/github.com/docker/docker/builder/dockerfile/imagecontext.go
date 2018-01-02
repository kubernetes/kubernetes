package dockerfile

import (
	"strconv"
	"strings"

	"github.com/docker/docker/api/types/backend"
	"github.com/docker/docker/builder"
	"github.com/docker/docker/builder/remotecontext"
	dockerimage "github.com/docker/docker/image"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
)

type buildStage struct {
	id string
}

func newBuildStage(imageID string) *buildStage {
	return &buildStage{id: imageID}
}

func (b *buildStage) ImageID() string {
	return b.id
}

func (b *buildStage) update(imageID string) {
	b.id = imageID
}

// buildStages tracks each stage of a build so they can be retrieved by index
// or by name.
type buildStages struct {
	sequence []*buildStage
	byName   map[string]*buildStage
}

func newBuildStages() *buildStages {
	return &buildStages{byName: make(map[string]*buildStage)}
}

func (s *buildStages) getByName(name string) (*buildStage, bool) {
	stage, ok := s.byName[strings.ToLower(name)]
	return stage, ok
}

func (s *buildStages) get(indexOrName string) (*buildStage, error) {
	index, err := strconv.Atoi(indexOrName)
	if err == nil {
		if err := s.validateIndex(index); err != nil {
			return nil, err
		}
		return s.sequence[index], nil
	}
	if im, ok := s.byName[strings.ToLower(indexOrName)]; ok {
		return im, nil
	}
	return nil, nil
}

func (s *buildStages) validateIndex(i int) error {
	if i < 0 || i >= len(s.sequence)-1 {
		if i == len(s.sequence)-1 {
			return errors.New("refers to current build stage")
		}
		return errors.New("index out of bounds")
	}
	return nil
}

func (s *buildStages) add(name string, image builder.Image) error {
	stage := newBuildStage(image.ImageID())
	name = strings.ToLower(name)
	if len(name) > 0 {
		if _, ok := s.byName[name]; ok {
			return errors.Errorf("duplicate name %s", name)
		}
		s.byName[name] = stage
	}
	s.sequence = append(s.sequence, stage)
	return nil
}

func (s *buildStages) update(imageID string) {
	s.sequence[len(s.sequence)-1].update(imageID)
}

type getAndMountFunc func(string, bool) (builder.Image, builder.ReleaseableLayer, error)

// imageSources mounts images and provides a cache for mounted images. It tracks
// all images so they can be unmounted at the end of the build.
type imageSources struct {
	byImageID map[string]*imageMount
	mounts    []*imageMount
	getImage  getAndMountFunc
	cache     pathCache // TODO: remove
}

// TODO @jhowardmsft LCOW Support: Eventually, platform can be moved to options.Options.Platform,
// and removed from builderOptions, but that can't be done yet as it would affect the API.
func newImageSources(ctx context.Context, options builderOptions) *imageSources {
	getAndMount := func(idOrRef string, localOnly bool) (builder.Image, builder.ReleaseableLayer, error) {
		pullOption := backend.PullOptionNoPull
		if !localOnly {
			if options.Options.PullParent {
				pullOption = backend.PullOptionForcePull
			} else {
				pullOption = backend.PullOptionPreferLocal
			}
		}
		return options.Backend.GetImageAndReleasableLayer(ctx, idOrRef, backend.GetImageAndLayerOptions{
			PullOption: pullOption,
			AuthConfig: options.Options.AuthConfigs,
			Output:     options.ProgressWriter.Output,
			Platform:   options.Platform,
		})
	}

	return &imageSources{
		byImageID: make(map[string]*imageMount),
		getImage:  getAndMount,
	}
}

func (m *imageSources) Get(idOrRef string, localOnly bool) (*imageMount, error) {
	if im, ok := m.byImageID[idOrRef]; ok {
		return im, nil
	}

	image, layer, err := m.getImage(idOrRef, localOnly)
	if err != nil {
		return nil, err
	}
	im := newImageMount(image, layer)
	m.Add(im)
	return im, nil
}

func (m *imageSources) Unmount() (retErr error) {
	for _, im := range m.mounts {
		if err := im.unmount(); err != nil {
			logrus.Error(err)
			retErr = err
		}
	}
	return
}

func (m *imageSources) Add(im *imageMount) {
	switch im.image {
	case nil:
		im.image = &dockerimage.Image{}
	default:
		m.byImageID[im.image.ImageID()] = im
	}
	m.mounts = append(m.mounts, im)
}

// imageMount is a reference to an image that can be used as a builder.Source
type imageMount struct {
	image  builder.Image
	source builder.Source
	layer  builder.ReleaseableLayer
}

func newImageMount(image builder.Image, layer builder.ReleaseableLayer) *imageMount {
	im := &imageMount{image: image, layer: layer}
	return im
}

func (im *imageMount) Source() (builder.Source, error) {
	if im.source == nil {
		if im.layer == nil {
			return nil, errors.Errorf("empty context")
		}
		mountPath, err := im.layer.Mount()
		if err != nil {
			return nil, errors.Wrapf(err, "failed to mount %s", im.image.ImageID())
		}
		source, err := remotecontext.NewLazySource(mountPath)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to create lazycontext for %s", mountPath)
		}
		im.source = source
	}
	return im.source, nil
}

func (im *imageMount) unmount() error {
	if im.layer == nil {
		return nil
	}
	if err := im.layer.Release(); err != nil {
		return errors.Wrapf(err, "failed to unmount previous build image %s", im.image.ImageID())
	}
	im.layer = nil
	return nil
}

func (im *imageMount) Image() builder.Image {
	return im.image
}

func (im *imageMount) Layer() builder.ReleaseableLayer {
	return im.layer
}

func (im *imageMount) ImageID() string {
	return im.image.ImageID()
}
