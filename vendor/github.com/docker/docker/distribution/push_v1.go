package distribution

import (
	"fmt"
	"sync"

	"github.com/docker/distribution/reference"
	"github.com/docker/distribution/registry/client/transport"
	"github.com/docker/docker/distribution/metadata"
	"github.com/docker/docker/dockerversion"
	"github.com/docker/docker/image"
	"github.com/docker/docker/image/v1"
	"github.com/docker/docker/layer"
	"github.com/docker/docker/pkg/ioutils"
	"github.com/docker/docker/pkg/progress"
	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/docker/registry"
	"github.com/opencontainers/go-digest"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
)

type v1Pusher struct {
	v1IDService *metadata.V1IDService
	endpoint    registry.APIEndpoint
	ref         reference.Named
	repoInfo    *registry.RepositoryInfo
	config      *ImagePushConfig
	session     *registry.Session
}

func (p *v1Pusher) Push(ctx context.Context) error {
	tlsConfig, err := p.config.RegistryService.TLSConfig(p.repoInfo.Index.Name)
	if err != nil {
		return err
	}
	// Adds Docker-specific headers as well as user-specified headers (metaHeaders)
	tr := transport.NewTransport(
		// TODO(tiborvass): was NoTimeout
		registry.NewTransport(tlsConfig),
		registry.DockerHeaders(dockerversion.DockerUserAgent(ctx), p.config.MetaHeaders)...,
	)
	client := registry.HTTPClient(tr)
	v1Endpoint, err := p.endpoint.ToV1Endpoint(dockerversion.DockerUserAgent(ctx), p.config.MetaHeaders)
	if err != nil {
		logrus.Debugf("Could not get v1 endpoint: %v", err)
		return fallbackError{err: err}
	}
	p.session, err = registry.NewSession(client, p.config.AuthConfig, v1Endpoint)
	if err != nil {
		// TODO(dmcgowan): Check if should fallback
		return fallbackError{err: err}
	}
	if err := p.pushRepository(ctx); err != nil {
		// TODO(dmcgowan): Check if should fallback
		return err
	}
	return nil
}

// v1Image exposes the configuration, filesystem layer ID, and a v1 ID for an
// image being pushed to a v1 registry.
type v1Image interface {
	Config() []byte
	Layer() layer.Layer
	V1ID() string
}

type v1ImageCommon struct {
	layer  layer.Layer
	config []byte
	v1ID   string
}

func (common *v1ImageCommon) Config() []byte {
	return common.config
}

func (common *v1ImageCommon) V1ID() string {
	return common.v1ID
}

func (common *v1ImageCommon) Layer() layer.Layer {
	return common.layer
}

// v1TopImage defines a runnable (top layer) image being pushed to a v1
// registry.
type v1TopImage struct {
	v1ImageCommon
	imageID image.ID
}

func newV1TopImage(imageID image.ID, img *image.Image, l layer.Layer, parent *v1DependencyImage) (*v1TopImage, error) {
	v1ID := imageID.Digest().Hex()
	parentV1ID := ""
	if parent != nil {
		parentV1ID = parent.V1ID()
	}

	config, err := v1.MakeV1ConfigFromConfig(img, v1ID, parentV1ID, false)
	if err != nil {
		return nil, err
	}

	return &v1TopImage{
		v1ImageCommon: v1ImageCommon{
			v1ID:   v1ID,
			config: config,
			layer:  l,
		},
		imageID: imageID,
	}, nil
}

// v1DependencyImage defines a dependency layer being pushed to a v1 registry.
type v1DependencyImage struct {
	v1ImageCommon
}

func newV1DependencyImage(l layer.Layer, parent *v1DependencyImage) *v1DependencyImage {
	v1ID := digest.Digest(l.ChainID()).Hex()

	var config string
	if parent != nil {
		config = fmt.Sprintf(`{"id":"%s","parent":"%s"}`, v1ID, parent.V1ID())
	} else {
		config = fmt.Sprintf(`{"id":"%s"}`, v1ID)
	}
	return &v1DependencyImage{
		v1ImageCommon: v1ImageCommon{
			v1ID:   v1ID,
			config: []byte(config),
			layer:  l,
		},
	}
}

// Retrieve the all the images to be uploaded in the correct order
func (p *v1Pusher) getImageList() (imageList []v1Image, tagsByImage map[image.ID][]string, referencedLayers []PushLayer, err error) {
	tagsByImage = make(map[image.ID][]string)

	// Ignore digest references
	if _, isCanonical := p.ref.(reference.Canonical); isCanonical {
		return
	}

	tagged, isTagged := p.ref.(reference.NamedTagged)
	if isTagged {
		// Push a specific tag
		var imgID image.ID
		var dgst digest.Digest
		dgst, err = p.config.ReferenceStore.Get(p.ref)
		if err != nil {
			return
		}
		imgID = image.IDFromDigest(dgst)

		imageList, err = p.imageListForTag(imgID, nil, &referencedLayers)
		if err != nil {
			return
		}

		tagsByImage[imgID] = []string{tagged.Tag()}

		return
	}

	imagesSeen := make(map[digest.Digest]struct{})
	dependenciesSeen := make(map[layer.ChainID]*v1DependencyImage)

	associations := p.config.ReferenceStore.ReferencesByName(p.ref)
	for _, association := range associations {
		if tagged, isTagged = association.Ref.(reference.NamedTagged); !isTagged {
			// Ignore digest references.
			continue
		}

		imgID := image.IDFromDigest(association.ID)
		tagsByImage[imgID] = append(tagsByImage[imgID], tagged.Tag())

		if _, present := imagesSeen[association.ID]; present {
			// Skip generating image list for already-seen image
			continue
		}
		imagesSeen[association.ID] = struct{}{}

		imageListForThisTag, err := p.imageListForTag(imgID, dependenciesSeen, &referencedLayers)
		if err != nil {
			return nil, nil, nil, err
		}

		// append to main image list
		imageList = append(imageList, imageListForThisTag...)
	}
	if len(imageList) == 0 {
		return nil, nil, nil, fmt.Errorf("No images found for the requested repository / tag")
	}
	logrus.Debugf("Image list: %v", imageList)
	logrus.Debugf("Tags by image: %v", tagsByImage)

	return
}

func (p *v1Pusher) imageListForTag(imgID image.ID, dependenciesSeen map[layer.ChainID]*v1DependencyImage, referencedLayers *[]PushLayer) (imageListForThisTag []v1Image, err error) {
	ics, ok := p.config.ImageStore.(*imageConfigStore)
	if !ok {
		return nil, fmt.Errorf("only image store images supported for v1 push")
	}
	img, err := ics.Store.Get(imgID)
	if err != nil {
		return nil, err
	}

	topLayerID := img.RootFS.ChainID()

	pl, err := p.config.LayerStore.Get(topLayerID)
	*referencedLayers = append(*referencedLayers, pl)
	if err != nil {
		return nil, fmt.Errorf("failed to get top layer from image: %v", err)
	}

	// V1 push is deprecated, only support existing layerstore layers
	lsl, ok := pl.(*storeLayer)
	if !ok {
		return nil, fmt.Errorf("only layer store layers supported for v1 push")
	}
	l := lsl.Layer

	dependencyImages, parent := generateDependencyImages(l.Parent(), dependenciesSeen)

	topImage, err := newV1TopImage(imgID, img, l, parent)
	if err != nil {
		return nil, err
	}

	imageListForThisTag = append(dependencyImages, topImage)

	return
}

func generateDependencyImages(l layer.Layer, dependenciesSeen map[layer.ChainID]*v1DependencyImage) (imageListForThisTag []v1Image, parent *v1DependencyImage) {
	if l == nil {
		return nil, nil
	}

	imageListForThisTag, parent = generateDependencyImages(l.Parent(), dependenciesSeen)

	if dependenciesSeen != nil {
		if dependencyImage, present := dependenciesSeen[l.ChainID()]; present {
			// This layer is already on the list, we can ignore it
			// and all its parents.
			return imageListForThisTag, dependencyImage
		}
	}

	dependencyImage := newV1DependencyImage(l, parent)
	imageListForThisTag = append(imageListForThisTag, dependencyImage)

	if dependenciesSeen != nil {
		dependenciesSeen[l.ChainID()] = dependencyImage
	}

	return imageListForThisTag, dependencyImage
}

// createImageIndex returns an index of an image's layer IDs and tags.
func createImageIndex(images []v1Image, tags map[image.ID][]string) []*registry.ImgData {
	var imageIndex []*registry.ImgData
	for _, img := range images {
		v1ID := img.V1ID()

		if topImage, isTopImage := img.(*v1TopImage); isTopImage {
			if tags, hasTags := tags[topImage.imageID]; hasTags {
				// If an image has tags you must add an entry in the image index
				// for each tag
				for _, tag := range tags {
					imageIndex = append(imageIndex, &registry.ImgData{
						ID:  v1ID,
						Tag: tag,
					})
				}
				continue
			}
		}

		// If the image does not have a tag it still needs to be sent to the
		// registry with an empty tag so that it is associated with the repository
		imageIndex = append(imageIndex, &registry.ImgData{
			ID:  v1ID,
			Tag: "",
		})
	}
	return imageIndex
}

// lookupImageOnEndpoint checks the specified endpoint to see if an image exists
// and if it is absent then it sends the image id to the channel to be pushed.
func (p *v1Pusher) lookupImageOnEndpoint(wg *sync.WaitGroup, endpoint string, images chan v1Image, imagesToPush chan string) {
	defer wg.Done()
	for image := range images {
		v1ID := image.V1ID()
		truncID := stringid.TruncateID(image.Layer().DiffID().String())
		if err := p.session.LookupRemoteImage(v1ID, endpoint); err != nil {
			logrus.Errorf("Error in LookupRemoteImage: %s", err)
			imagesToPush <- v1ID
			progress.Update(p.config.ProgressOutput, truncID, "Waiting")
		} else {
			progress.Update(p.config.ProgressOutput, truncID, "Already exists")
		}
	}
}

func (p *v1Pusher) pushImageToEndpoint(ctx context.Context, endpoint string, imageList []v1Image, tags map[image.ID][]string, repo *registry.RepositoryData) error {
	workerCount := len(imageList)
	// start a maximum of 5 workers to check if images exist on the specified endpoint.
	if workerCount > 5 {
		workerCount = 5
	}
	var (
		wg           = &sync.WaitGroup{}
		imageData    = make(chan v1Image, workerCount*2)
		imagesToPush = make(chan string, workerCount*2)
		pushes       = make(chan map[string]struct{}, 1)
	)
	for i := 0; i < workerCount; i++ {
		wg.Add(1)
		go p.lookupImageOnEndpoint(wg, endpoint, imageData, imagesToPush)
	}
	// start a go routine that consumes the images to push
	go func() {
		shouldPush := make(map[string]struct{})
		for id := range imagesToPush {
			shouldPush[id] = struct{}{}
		}
		pushes <- shouldPush
	}()
	for _, v1Image := range imageList {
		imageData <- v1Image
	}
	// close the channel to notify the workers that there will be no more images to check.
	close(imageData)
	wg.Wait()
	close(imagesToPush)
	// wait for all the images that require pushes to be collected into a consumable map.
	shouldPush := <-pushes
	// finish by pushing any images and tags to the endpoint.  The order that the images are pushed
	// is very important that is why we are still iterating over the ordered list of imageIDs.
	for _, img := range imageList {
		v1ID := img.V1ID()
		if _, push := shouldPush[v1ID]; push {
			if _, err := p.pushImage(ctx, img, endpoint); err != nil {
				// FIXME: Continue on error?
				return err
			}
		}
		if topImage, isTopImage := img.(*v1TopImage); isTopImage {
			for _, tag := range tags[topImage.imageID] {
				progress.Messagef(p.config.ProgressOutput, "", "Pushing tag for rev [%s] on {%s}", stringid.TruncateID(v1ID), endpoint+"repositories/"+reference.Path(p.repoInfo.Name)+"/tags/"+tag)
				if err := p.session.PushRegistryTag(p.repoInfo.Name, v1ID, tag, endpoint); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// pushRepository pushes layers that do not already exist on the registry.
func (p *v1Pusher) pushRepository(ctx context.Context) error {
	imgList, tags, referencedLayers, err := p.getImageList()
	defer func() {
		for _, l := range referencedLayers {
			l.Release()
		}
	}()
	if err != nil {
		return err
	}

	imageIndex := createImageIndex(imgList, tags)
	for _, data := range imageIndex {
		logrus.Debugf("Pushing ID: %s with Tag: %s", data.ID, data.Tag)
	}

	// Register all the images in a repository with the registry
	// If an image is not in this list it will not be associated with the repository
	repoData, err := p.session.PushImageJSONIndex(p.repoInfo.Name, imageIndex, false, nil)
	if err != nil {
		return err
	}
	// push the repository to each of the endpoints only if it does not exist.
	for _, endpoint := range repoData.Endpoints {
		if err := p.pushImageToEndpoint(ctx, endpoint, imgList, tags, repoData); err != nil {
			return err
		}
	}
	_, err = p.session.PushImageJSONIndex(p.repoInfo.Name, imageIndex, true, repoData.Endpoints)
	return err
}

func (p *v1Pusher) pushImage(ctx context.Context, v1Image v1Image, ep string) (checksum string, err error) {
	l := v1Image.Layer()
	v1ID := v1Image.V1ID()
	truncID := stringid.TruncateID(l.DiffID().String())

	jsonRaw := v1Image.Config()
	progress.Update(p.config.ProgressOutput, truncID, "Pushing")

	// General rule is to use ID for graph accesses and compatibilityID for
	// calls to session.registry()
	imgData := &registry.ImgData{
		ID: v1ID,
	}

	// Send the json
	if err := p.session.PushImageJSONRegistry(imgData, jsonRaw, ep); err != nil {
		if err == registry.ErrAlreadyExists {
			progress.Update(p.config.ProgressOutput, truncID, "Image already pushed, skipping")
			return "", nil
		}
		return "", err
	}

	arch, err := l.TarStream()
	if err != nil {
		return "", err
	}
	defer arch.Close()

	// don't care if this fails; best effort
	size, _ := l.DiffSize()

	// Send the layer
	logrus.Debugf("rendered layer for %s of [%d] size", v1ID, size)

	reader := progress.NewProgressReader(ioutils.NewCancelReadCloser(ctx, arch), p.config.ProgressOutput, size, truncID, "Pushing")
	defer reader.Close()

	checksum, checksumPayload, err := p.session.PushImageLayerRegistry(v1ID, reader, ep, jsonRaw)
	if err != nil {
		return "", err
	}
	imgData.Checksum = checksum
	imgData.ChecksumPayload = checksumPayload
	// Send the checksum
	if err := p.session.PushImageChecksumRegistry(imgData, ep); err != nil {
		return "", err
	}

	if err := p.v1IDService.Set(v1ID, p.repoInfo.Index.Name, l.DiffID()); err != nil {
		logrus.Warnf("Could not set v1 ID mapping: %v", err)
	}

	progress.Update(p.config.ProgressOutput, truncID, "Image successfully pushed")
	return imgData.Checksum, nil
}
