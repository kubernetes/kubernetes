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

package crane

import "github.com/google/go-containerregistry/pkg/logs"

// Digest returns the sha256 hash of the remote image at ref.
func Digest(ref string, opt ...Option) (string, error) {
	o := makeOptions(opt...)
	if o.Platform != nil {
		desc, err := getManifest(ref, opt...)
		if err != nil {
			return "", err
		}
		if !desc.MediaType.IsIndex() {
			return desc.Digest.String(), nil
		}

		// TODO: does not work for indexes which contain schema v1 manifests
		img, err := desc.Image()
		if err != nil {
			return "", err
		}
		digest, err := img.Digest()
		if err != nil {
			return "", err
		}
		return digest.String(), nil
	}
	desc, err := Head(ref, opt...)
	if err != nil {
		logs.Warn.Printf("HEAD request failed, falling back on GET: %v", err)
		rdesc, err := getManifest(ref, opt...)
		if err != nil {
			return "", err
		}
		return rdesc.Digest.String(), nil
	}
	return desc.Digest.String(), nil
}
