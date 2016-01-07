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

package lastditch

import (
	"encoding/json"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
)

type PodManifest struct {
	ACVersion string  `json:"acVersion"`
	ACKind    string  `json:"acKind"`
	Apps      AppList `json:"apps"`
}

type AppList []RuntimeApp

type RuntimeApp struct {
	Name  string       `json:"name"`
	Image RuntimeImage `json:"image"`
}

type RuntimeImage struct {
	Name   string `json:"name"`
	ID     string `json:"id"`
	Labels Labels `json:"labels,omitempty"`
}

// a type just to avoid a recursion during unmarshalling
type podManifest PodManifest

func (pm *PodManifest) UnmarshalJSON(data []byte) error {
	p := podManifest(*pm)
	err := json.Unmarshal(data, &p)
	if err != nil {
		return err
	}
	if p.ACKind != string(schema.PodManifestKind) {
		return types.InvalidACKindError(schema.PodManifestKind)
	}
	*pm = PodManifest(p)
	return nil
}
