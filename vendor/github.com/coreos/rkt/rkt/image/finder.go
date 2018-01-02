// Copyright 2015 The rkt Authors
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

package image

import (
	"fmt"

	"github.com/coreos/rkt/common/apps"
	"github.com/hashicorp/errwrap"

	"github.com/appc/spec/schema/types"
)

// Finder will try to get images from the store. If not found, it will
// try to fetch them.
type Finder action

// FindImages uses FindImage to attain a list of image hashes
func (f *Finder) FindImages(al *apps.Apps) error {
	return al.Walk(func(app *apps.App) error {
		h, err := f.FindImage(app.Image, app.Asc)
		if err != nil {
			return err
		}
		app.ImageID = *h
		return nil
	})
}

// FindImage tries to get a hash of a passed image, ideally from
// store. Otherwise this might involve fetching it from remote with
// the Fetcher.
func (f *Finder) FindImage(img string, asc string) (*types.Hash, error) {
	ensureLogger(f.Debug)

	// Check if it's an hash
	if _, err := types.NewHash(img); err == nil {
		h, err := f.getHashFromStore(img)
		if err != nil {
			return nil, err
		}
		return h, nil
	}

	d, err := DistFromImageString(img)
	if err != nil {
		return nil, err
	}

	// urls, names, paths have to be fetched, potentially remotely
	ft := (*Fetcher)(f)
	h, err := ft.FetchImage(d, img, asc)
	if err != nil {
		return nil, err
	}
	return h, nil
}

func (f *Finder) getHashFromStore(img string) (*types.Hash, error) {
	h, err := types.NewHash(img)
	if err != nil {
		return nil, errwrap.Wrap(fmt.Errorf("%q is not a valid hash", img), err)
	}
	fullKey, err := f.S.ResolveKey(img)
	if err != nil {
		return nil, errwrap.Wrap(fmt.Errorf("could not resolve image %q", img), err)
	}
	h, err = types.NewHash(fullKey)
	if err != nil {
		// should never happen
		log.PanicE("got an invalid hash from the store, looks like it is corrupted", err)
	}
	diag.Printf("using image from the store with hash %s", h.String())
	return h, nil
}
