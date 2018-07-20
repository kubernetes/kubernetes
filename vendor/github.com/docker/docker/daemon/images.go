package daemon

import (
	"encoding/json"
	"fmt"
	"runtime"
	"sort"
	"time"

	"github.com/pkg/errors"

	"github.com/docker/distribution/reference"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/filters"
	"github.com/docker/docker/container"
	"github.com/docker/docker/image"
	"github.com/docker/docker/layer"
	"github.com/docker/docker/pkg/system"
)

var acceptedImageFilterTags = map[string]bool{
	"dangling":  true,
	"label":     true,
	"before":    true,
	"since":     true,
	"reference": true,
}

// byCreated is a temporary type used to sort a list of images by creation
// time.
type byCreated []*types.ImageSummary

func (r byCreated) Len() int           { return len(r) }
func (r byCreated) Swap(i, j int)      { r[i], r[j] = r[j], r[i] }
func (r byCreated) Less(i, j int) bool { return r[i].Created < r[j].Created }

// Map returns a map of all images in the ImageStore
func (daemon *Daemon) Map() map[image.ID]*image.Image {
	// TODO @jhowardmsft LCOW. This can be removed when imagestores are coalesced
	platform := runtime.GOOS
	if system.LCOWSupported() {
		platform = "linux"
	}
	return daemon.stores[platform].imageStore.Map()
}

// Images returns a filtered list of images. filterArgs is a JSON-encoded set
// of filter arguments which will be interpreted by api/types/filters.
// filter is a shell glob string applied to repository names. The argument
// named all controls whether all images in the graph are filtered, or just
// the heads.
func (daemon *Daemon) Images(imageFilters filters.Args, all bool, withExtraAttrs bool) ([]*types.ImageSummary, error) {

	// TODO @jhowardmsft LCOW. This can be removed when imagestores are coalesced
	platform := runtime.GOOS
	if system.LCOWSupported() {
		platform = "linux"
	}

	var (
		allImages    map[image.ID]*image.Image
		err          error
		danglingOnly = false
	)

	if err := imageFilters.Validate(acceptedImageFilterTags); err != nil {
		return nil, err
	}

	if imageFilters.Contains("dangling") {
		if imageFilters.ExactMatch("dangling", "true") {
			danglingOnly = true
		} else if !imageFilters.ExactMatch("dangling", "false") {
			return nil, invalidFilter{"dangling", imageFilters.Get("dangling")}
		}
	}
	if danglingOnly {
		allImages = daemon.stores[platform].imageStore.Heads()
	} else {
		allImages = daemon.stores[platform].imageStore.Map()
	}

	var beforeFilter, sinceFilter *image.Image
	err = imageFilters.WalkValues("before", func(value string) error {
		beforeFilter, err = daemon.GetImage(value)
		return err
	})
	if err != nil {
		return nil, err
	}

	err = imageFilters.WalkValues("since", func(value string) error {
		sinceFilter, err = daemon.GetImage(value)
		return err
	})
	if err != nil {
		return nil, err
	}

	images := []*types.ImageSummary{}
	var imagesMap map[*image.Image]*types.ImageSummary
	var layerRefs map[layer.ChainID]int
	var allLayers map[layer.ChainID]layer.Layer
	var allContainers []*container.Container

	for id, img := range allImages {
		if beforeFilter != nil {
			if img.Created.Equal(beforeFilter.Created) || img.Created.After(beforeFilter.Created) {
				continue
			}
		}

		if sinceFilter != nil {
			if img.Created.Equal(sinceFilter.Created) || img.Created.Before(sinceFilter.Created) {
				continue
			}
		}

		if imageFilters.Contains("label") {
			// Very old image that do not have image.Config (or even labels)
			if img.Config == nil {
				continue
			}
			// We are now sure image.Config is not nil
			if !imageFilters.MatchKVList("label", img.Config.Labels) {
				continue
			}
		}

		layerID := img.RootFS.ChainID()
		var size int64
		if layerID != "" {
			l, err := daemon.stores[platform].layerStore.Get(layerID)
			if err != nil {
				// The layer may have been deleted between the call to `Map()` or
				// `Heads()` and the call to `Get()`, so we just ignore this error
				if err == layer.ErrLayerDoesNotExist {
					continue
				}
				return nil, err
			}

			size, err = l.Size()
			layer.ReleaseAndLog(daemon.stores[platform].layerStore, l)
			if err != nil {
				return nil, err
			}
		}

		newImage := newImage(img, size)

		for _, ref := range daemon.referenceStore.References(id.Digest()) {
			if imageFilters.Contains("reference") {
				var found bool
				var matchErr error
				for _, pattern := range imageFilters.Get("reference") {
					found, matchErr = reference.FamiliarMatch(pattern, ref)
					if matchErr != nil {
						return nil, matchErr
					}
				}
				if !found {
					continue
				}
			}
			if _, ok := ref.(reference.Canonical); ok {
				newImage.RepoDigests = append(newImage.RepoDigests, reference.FamiliarString(ref))
			}
			if _, ok := ref.(reference.NamedTagged); ok {
				newImage.RepoTags = append(newImage.RepoTags, reference.FamiliarString(ref))
			}
		}
		if newImage.RepoDigests == nil && newImage.RepoTags == nil {
			if all || len(daemon.stores[platform].imageStore.Children(id)) == 0 {

				if imageFilters.Contains("dangling") && !danglingOnly {
					//dangling=false case, so dangling image is not needed
					continue
				}
				if imageFilters.Contains("reference") { // skip images with no references if filtering by reference
					continue
				}
				newImage.RepoDigests = []string{"<none>@<none>"}
				newImage.RepoTags = []string{"<none>:<none>"}
			} else {
				continue
			}
		} else if danglingOnly && len(newImage.RepoTags) > 0 {
			continue
		}

		if withExtraAttrs {
			// lazily init variables
			if imagesMap == nil {
				allContainers = daemon.List()
				allLayers = daemon.stores[platform].layerStore.Map()
				imagesMap = make(map[*image.Image]*types.ImageSummary)
				layerRefs = make(map[layer.ChainID]int)
			}

			// Get container count
			newImage.Containers = 0
			for _, c := range allContainers {
				if c.ImageID == id {
					newImage.Containers++
				}
			}

			// count layer references
			rootFS := *img.RootFS
			rootFS.DiffIDs = nil
			for _, id := range img.RootFS.DiffIDs {
				rootFS.Append(id)
				chid := rootFS.ChainID()
				layerRefs[chid]++
				if _, ok := allLayers[chid]; !ok {
					return nil, fmt.Errorf("layer %v was not found (corruption?)", chid)
				}
			}
			imagesMap[img] = newImage
		}

		images = append(images, newImage)
	}

	if withExtraAttrs {
		// Get Shared sizes
		for img, newImage := range imagesMap {
			rootFS := *img.RootFS
			rootFS.DiffIDs = nil

			newImage.SharedSize = 0
			for _, id := range img.RootFS.DiffIDs {
				rootFS.Append(id)
				chid := rootFS.ChainID()

				diffSize, err := allLayers[chid].DiffSize()
				if err != nil {
					return nil, err
				}

				if layerRefs[chid] > 1 {
					newImage.SharedSize += diffSize
				}
			}
		}
	}

	sort.Sort(sort.Reverse(byCreated(images)))

	return images, nil
}

// SquashImage creates a new image with the diff of the specified image and the specified parent.
// This new image contains only the layers from it's parent + 1 extra layer which contains the diff of all the layers in between.
// The existing image(s) is not destroyed.
// If no parent is specified, a new image with the diff of all the specified image's layers merged into a new layer that has no parents.
func (daemon *Daemon) SquashImage(id, parent string) (string, error) {

	var (
		img *image.Image
		err error
	)
	for _, ds := range daemon.stores {
		if img, err = ds.imageStore.Get(image.ID(id)); err == nil {
			break
		}
	}
	if err != nil {
		return "", err
	}

	var parentImg *image.Image
	var parentChainID layer.ChainID
	if len(parent) != 0 {
		parentImg, err = daemon.stores[img.OperatingSystem()].imageStore.Get(image.ID(parent))
		if err != nil {
			return "", errors.Wrap(err, "error getting specified parent layer")
		}
		parentChainID = parentImg.RootFS.ChainID()
	} else {
		rootFS := image.NewRootFS()
		parentImg = &image.Image{RootFS: rootFS}
	}

	l, err := daemon.stores[img.OperatingSystem()].layerStore.Get(img.RootFS.ChainID())
	if err != nil {
		return "", errors.Wrap(err, "error getting image layer")
	}
	defer daemon.stores[img.OperatingSystem()].layerStore.Release(l)

	ts, err := l.TarStreamFrom(parentChainID)
	if err != nil {
		return "", errors.Wrapf(err, "error getting tar stream to parent")
	}
	defer ts.Close()

	newL, err := daemon.stores[img.OperatingSystem()].layerStore.Register(ts, parentChainID, layer.OS(img.OperatingSystem()))
	if err != nil {
		return "", errors.Wrap(err, "error registering layer")
	}
	defer daemon.stores[img.OperatingSystem()].layerStore.Release(newL)

	newImage := *img
	newImage.RootFS = nil

	rootFS := *parentImg.RootFS
	rootFS.DiffIDs = append(rootFS.DiffIDs, newL.DiffID())
	newImage.RootFS = &rootFS

	for i, hi := range newImage.History {
		if i >= len(parentImg.History) {
			hi.EmptyLayer = true
		}
		newImage.History[i] = hi
	}

	now := time.Now()
	var historyComment string
	if len(parent) > 0 {
		historyComment = fmt.Sprintf("merge %s to %s", id, parent)
	} else {
		historyComment = fmt.Sprintf("create new from %s", id)
	}

	newImage.History = append(newImage.History, image.History{
		Created: now,
		Comment: historyComment,
	})
	newImage.Created = now

	b, err := json.Marshal(&newImage)
	if err != nil {
		return "", errors.Wrap(err, "error marshalling image config")
	}

	newImgID, err := daemon.stores[img.OperatingSystem()].imageStore.Create(b)
	if err != nil {
		return "", errors.Wrap(err, "error creating new image after squash")
	}
	return string(newImgID), nil
}

func newImage(image *image.Image, size int64) *types.ImageSummary {
	newImage := new(types.ImageSummary)
	newImage.ParentID = image.Parent.String()
	newImage.ID = image.ID().String()
	newImage.Created = image.Created.Unix()
	newImage.Size = size
	newImage.VirtualSize = size
	newImage.SharedSize = -1
	newImage.Containers = -1
	if image.Config != nil {
		newImage.Labels = image.Config.Labels
	}
	return newImage
}
