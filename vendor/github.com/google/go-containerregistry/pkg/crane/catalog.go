// Copyright 2019 Google LLC All Rights Reserved.
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
	"context"

	"github.com/google/go-containerregistry/pkg/name"
	"github.com/google/go-containerregistry/pkg/v1/remote"
)

// Catalog returns the repositories in a registry's catalog.
func Catalog(src string, opt ...Option) (res []string, err error) {
	o := makeOptions(opt...)
	reg, err := name.NewRegistry(src, o.Name...)
	if err != nil {
		return nil, err
	}

	// This context gets overridden by remote.WithContext, which is set by
	// crane.WithContext.
	return remote.Catalog(context.Background(), reg, o.Remote...)
}
