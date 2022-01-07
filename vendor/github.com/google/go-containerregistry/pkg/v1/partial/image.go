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

package partial

import (
	"github.com/google/go-containerregistry/pkg/v1/types"
)

// ImageCore is the core set of properties without which we cannot build a v1.Image
type ImageCore interface {
	// RawConfigFile returns the serialized bytes of this image's config file.
	RawConfigFile() ([]byte, error)

	// MediaType of this image's manifest.
	MediaType() (types.MediaType, error)
}
