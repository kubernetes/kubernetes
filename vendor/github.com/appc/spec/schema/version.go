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
	"github.com/appc/spec/schema/types"
)

const (
	// version represents the canonical version of the appc spec and tooling.
	// For now, the schema and tooling is coupled with the spec itself, so
	// this must be kept in sync with the VERSION file in the root of the repo.
	version string = "0.8.1+git"
)

var (
	// AppContainerVersion is the SemVer representation of version
	AppContainerVersion types.SemVer
)

func init() {
	v, err := types.NewSemVer(version)
	if err != nil {
		panic(err)
	}
	AppContainerVersion = *v
}
