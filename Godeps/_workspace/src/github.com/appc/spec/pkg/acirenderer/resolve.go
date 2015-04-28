package acirenderer

import (
	"container/list"

	"github.com/appc/spec/schema/types"
)

// CreateDepListFromImageID returns the flat dependency tree of the image with
// the provided imageID
func CreateDepListFromImageID(imageID types.Hash, ap ACIRegistry) (Images, error) {
	key, err := ap.ResolveKey(imageID.String())
	if err != nil {
		return nil, err
	}
	return createDepList(key, ap)
}

// CreateDepListFromNameLabels returns the flat dependency tree of the image
// with the provided app name and optional labels.
func CreateDepListFromNameLabels(name types.ACName, labels types.Labels, ap ACIRegistry) (Images, error) {
	key, err := ap.GetACI(name, labels)
	if err != nil {
		return nil, err
	}
	return createDepList(key, ap)
}

// createDepList returns the flat dependency tree as a list of Image type
func createDepList(key string, ap ACIRegistry) (Images, error) {
	imgsl := list.New()
	im, err := ap.GetImageManifest(key)
	if err != nil {
		return nil, err
	}

	img := Image{Im: im, Key: key, Level: 0}
	imgsl.PushFront(img)

	// Create a flat dependency tree. Use a LinkedList to be able to
	// insert elements in the list while working on it.
	for el := imgsl.Front(); el != nil; el = el.Next() {
		img := el.Value.(Image)
		dependencies := img.Im.Dependencies
		for _, d := range dependencies {
			var depimg Image
			var depKey string
			if d.ImageID != nil && !d.ImageID.Empty() {
				depKey, err = ap.ResolveKey(d.ImageID.String())
				if err != nil {
					return nil, err
				}
			} else {
				var err error
				depKey, err = ap.GetACI(d.App, d.Labels)
				if err != nil {
					return nil, err
				}
			}
			im, err := ap.GetImageManifest(depKey)
			if err != nil {
				return nil, err
			}
			depimg = Image{Im: im, Key: depKey, Level: img.Level + 1}
			imgsl.InsertAfter(depimg, el)
		}
	}

	imgs := Images{}
	for el := imgsl.Front(); el != nil; el = el.Next() {
		imgs = append(imgs, el.Value.(Image))
	}
	return imgs, nil
}
