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

package schema

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/appc/spec/schema/types"

	"go4.org/errorutil"
)

const PodManifestKind = types.ACKind("PodManifest")

type PodManifest struct {
	ACVersion       types.SemVer          `json:"acVersion"`
	ACKind          types.ACKind          `json:"acKind"`
	Apps            AppList               `json:"apps"`
	Volumes         []types.Volume        `json:"volumes"`
	Isolators       []types.Isolator      `json:"isolators"`
	Annotations     types.Annotations     `json:"annotations"`
	Ports           []types.ExposedPort   `json:"ports"`
	UserAnnotations types.UserAnnotations `json:"userAnnotations,omitempty"`
	UserLabels      types.UserLabels      `json:"userLabels,omitempty"`
}

// podManifest is a model to facilitate extra validation during the
// unmarshalling of the PodManifest
type podManifest PodManifest

func BlankPodManifest() *PodManifest {
	return &PodManifest{ACKind: PodManifestKind, ACVersion: AppContainerVersion}
}

func (pm *PodManifest) UnmarshalJSON(data []byte) error {
	p := podManifest(*pm)
	err := json.Unmarshal(data, &p)
	if err != nil {
		if serr, ok := err.(*json.SyntaxError); ok {
			line, col, highlight := errorutil.HighlightBytePosition(bytes.NewReader(data), serr.Offset)
			return fmt.Errorf("\nError at line %d, column %d\n%s%v", line, col, highlight, err)
		}
		return err
	}
	npm := PodManifest(p)
	if err := npm.assertValid(); err != nil {
		return err
	}
	*pm = npm
	return nil
}

func (pm PodManifest) MarshalJSON() ([]byte, error) {
	if err := pm.assertValid(); err != nil {
		return nil, err
	}
	return json.Marshal(podManifest(pm))
}

var pmKindError = types.InvalidACKindError(PodManifestKind)

// assertValid performs extra assertions on an PodManifest to
// ensure that fields are set appropriately, etc. It is used exclusively when
// marshalling and unmarshalling an PodManifest. Most
// field-specific validation is performed through the individual types being
// marshalled; assertValid() should only deal with higher-level validation.
func (pm *PodManifest) assertValid() error {
	if pm.ACKind != PodManifestKind {
		return pmKindError
	}
	return nil
}

type AppList []RuntimeApp

type appList AppList

func (al *AppList) UnmarshalJSON(data []byte) error {
	a := appList{}
	err := json.Unmarshal(data, &a)
	if err != nil {
		return err
	}
	nal := AppList(a)
	if err := nal.assertValid(); err != nil {
		return err
	}
	*al = nal
	return nil
}

func (al AppList) MarshalJSON() ([]byte, error) {
	if err := al.assertValid(); err != nil {
		return nil, err
	}
	return json.Marshal(appList(al))
}

func (al AppList) assertValid() error {
	seen := map[types.ACName]bool{}
	for _, a := range al {
		if _, ok := seen[a.Name]; ok {
			return fmt.Errorf(`duplicate apps of name %q`, a.Name)
		}
		seen[a.Name] = true
	}
	return nil
}

// Get retrieves an app by the specified name from the AppList; if there is
// no such app, nil is returned. The returned *RuntimeApp MUST be considered
// read-only.
func (al AppList) Get(name types.ACName) *RuntimeApp {
	for _, a := range al {
		if name.Equals(a.Name) {
			aa := a
			return &aa
		}
	}
	return nil
}

// Mount describes the mapping between a volume and the path it is mounted
// inside of an app's filesystem.
// The AppVolume is optional. If missing, the pod-level Volume of the
// same name shall be used.
type Mount struct {
	Volume    types.ACName  `json:"volume"`
	Path      string        `json:"path"`
	AppVolume *types.Volume `json:"appVolume,omitempty"`
}

func (r Mount) assertValid() error {
	if r.Volume.Empty() {
		return errors.New("volume must be set")
	}
	if r.Path == "" {
		return errors.New("path must be set")
	}
	return nil
}

// RuntimeApp describes an application referenced in a PodManifest
type RuntimeApp struct {
	Name           types.ACName      `json:"name"`
	Image          RuntimeImage      `json:"image"`
	App            *types.App        `json:"app,omitempty"`
	ReadOnlyRootFS bool              `json:"readOnlyRootFS,omitempty"`
	Mounts         []Mount           `json:"mounts,omitempty"`
	Annotations    types.Annotations `json:"annotations,omitempty"`
}

// RuntimeImage describes an image referenced in a RuntimeApp
type RuntimeImage struct {
	Name   *types.ACIdentifier `json:"name,omitempty"`
	ID     types.Hash          `json:"id"`
	Labels types.Labels        `json:"labels,omitempty"`
}
