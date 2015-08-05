// Copyright 2015 CoreOS, Inc.
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

//+build linux

package apps

import (
	"github.com/appc/spec/schema/types"
)

type App struct {
	Image string   // the image reference as supplied by the user on the cli
	Args  []string // any arguments the user supplied for this app
	Asc   string   // signature file override for image verification (if fetching occurs)

	// TODO(jonboulle): These images are partially-populated hashes, this should be clarified.
	ImageID types.Hash // resolved image identifier
}

type Apps struct {
	apps []App
}

// Reset creates a new slice for al.apps, needed by tests
func (al *Apps) Reset() {
	al.apps = make([]App, 0)
}

// Count returns the number of apps in al
func (al *Apps) Count() int {
	return len(al.apps)
}

// Create creates a new app in al and returns a pointer to it
func (al *Apps) Create(img string) {
	al.apps = append(al.apps, App{Image: img})
}

// Last returns a pointer to the top app in al
func (al *Apps) Last() *App {
	if len(al.apps) == 0 {
		return nil
	}
	return &al.apps[len(al.apps)-1]
}

// Walk iterates on al.apps calling f for each app
// walking stops if f returns an error, the error is simply returned
func (al *Apps) Walk(f func(*App) error) error {
	for i, _ := range al.apps {
		// XXX(vc): note we supply f() with a pointer to the app instance in al.apps to enable modification by f()
		if err := f(&al.apps[i]); err != nil {
			return err
		}
	}
	return nil
}

// these convenience functions just return typed lists containing just the named member
// TODO(vc): these probably go away when we just pass Apps to stage0

// GetImages returns a list of the images in al, one per app.
// The order reflects the app order in al.
func (al *Apps) GetImages() []string {
	il := []string{}
	for _, a := range al.apps {
		il = append(il, a.Image)
	}
	return il
}

// GetArgs returns a list of lists of arguments in al, one list of args per app.
// The order reflects the app order in al.
func (al *Apps) GetArgs() [][]string {
	aal := [][]string{}
	for _, a := range al.apps {
		aal = append(aal, a.Args)
	}
	return aal
}

// GetImageIDs returns a list of the imageIDs in al, one per app.
// The order reflects the app order in al.
func (al *Apps) GetImageIDs() []types.Hash {
	hl := []types.Hash{}
	for _, a := range al.apps {
		hl = append(hl, a.ImageID)
	}
	return hl
}
