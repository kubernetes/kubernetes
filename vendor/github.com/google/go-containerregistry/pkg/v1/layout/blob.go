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

package layout

import (
	"io"
	"os"

	v1 "github.com/google/go-containerregistry/pkg/v1"
)

// Blob returns a blob with the given hash from the Path.
func (l Path) Blob(h v1.Hash) (io.ReadCloser, error) {
	return os.Open(l.blobPath(h))
}

// Bytes is a convenience function to return a blob from the Path as
// a byte slice.
func (l Path) Bytes(h v1.Hash) ([]byte, error) {
	return os.ReadFile(l.blobPath(h))
}

func (l Path) blobPath(h v1.Hash) string {
	return l.path("blobs", h.Algorithm, h.Hex)
}
