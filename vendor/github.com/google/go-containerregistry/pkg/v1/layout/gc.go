// Copyright 2018 Google LLC All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This is an EXPERIMENTAL package, and may change in arbitrary ways without notice.
package layout

import (
	"fmt"
	"io/fs"
	"path/filepath"
	"strings"

	v1 "github.com/google/go-containerregistry/pkg/v1"
)

// GarbageCollect removes unreferenced blobs from the oci-layout
//
//	This is an experimental api, and not subject to any stability guarantees
//	We may abandon it at any time, without prior notice.
//	Deprecated: Use it at your own risk!
func (l Path) GarbageCollect() ([]v1.Hash, error) {
	idx, err := l.ImageIndex()
	if err != nil {
		return nil, err
	}
	blobsToKeep := map[string]bool{}
	if err := l.garbageCollectImageIndex(idx, blobsToKeep); err != nil {
		return nil, err
	}
	blobsDir := l.path("blobs")
	removedBlobs := []v1.Hash{}

	err = filepath.WalkDir(blobsDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		if d.IsDir() {
			return nil
		}

		rel, err := filepath.Rel(blobsDir, path)
		if err != nil {
			return err
		}
		hashString := strings.Replace(rel, "/", ":", 1)
		if present := blobsToKeep[hashString]; !present {
			h, err := v1.NewHash(hashString)
			if err != nil {
				return err
			}
			removedBlobs = append(removedBlobs, h)
		}
		return nil
	})

	if err != nil {
		return nil, err
	}

	return removedBlobs, nil
}

func (l Path) garbageCollectImageIndex(index v1.ImageIndex, blobsToKeep map[string]bool) error {
	idxm, err := index.IndexManifest()
	if err != nil {
		return err
	}

	h, err := index.Digest()
	if err != nil {
		return err
	}

	blobsToKeep[h.String()] = true

	for _, descriptor := range idxm.Manifests {
		if descriptor.MediaType.IsImage() {
			img, err := index.Image(descriptor.Digest)
			if err != nil {
				return err
			}
			if err := l.garbageCollectImage(img, blobsToKeep); err != nil {
				return err
			}
		} else if descriptor.MediaType.IsIndex() {
			idx, err := index.ImageIndex(descriptor.Digest)
			if err != nil {
				return err
			}
			if err := l.garbageCollectImageIndex(idx, blobsToKeep); err != nil {
				return err
			}
		} else {
			return fmt.Errorf("gc: unknown media type: %s", descriptor.MediaType)
		}
	}
	return nil
}

func (l Path) garbageCollectImage(image v1.Image, blobsToKeep map[string]bool) error {
	h, err := image.Digest()
	if err != nil {
		return err
	}
	blobsToKeep[h.String()] = true

	h, err = image.ConfigName()
	if err != nil {
		return err
	}
	blobsToKeep[h.String()] = true

	ls, err := image.Layers()
	if err != nil {
		return err
	}
	for _, l := range ls {
		h, err := l.Digest()
		if err != nil {
			return err
		}
		blobsToKeep[h.String()] = true
	}
	return nil
}
