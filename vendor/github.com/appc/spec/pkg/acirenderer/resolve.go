// Copyright 2015 The appc Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
func CreateDepListFromNameLabels(name types.ACIdentifier, labels types.Labels, ap ACIRegistry) (Images, error) {
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
				depKey, err = ap.GetACI(d.ImageName, d.Labels)
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
