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

import (
	"fmt"

	"github.com/google/go-containerregistry/pkg/name"
	"github.com/google/go-containerregistry/pkg/v1/remote"
)

// ListTags returns the tags in repository src.
func ListTags(src string, opt ...Option) ([]string, error) {
	o := makeOptions(opt...)
	repo, err := name.NewRepository(src, o.Name...)
	if err != nil {
		return nil, fmt.Errorf("parsing repo %q: %w", src, err)
	}

	return remote.List(repo, o.Remote...)
}
