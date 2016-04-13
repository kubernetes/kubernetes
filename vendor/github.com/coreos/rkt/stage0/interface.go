// Copyright 2016 The rkt Authors
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

package stage0

import (
	"encoding/json"
	"errors"
	"io/ioutil"
	"strconv"

	"github.com/appc/spec/schema"
	"github.com/coreos/rkt/common"
	"github.com/hashicorp/errwrap"
)

const (
	interfaceVersion = "coreos.com/rkt/stage1/interface-version"
)

// getStage1InterfaceVersion retrieves the interface version from the stage1
// manifest for a given pod
func getStage1InterfaceVersion(cdir string) (int, error) {
	b, err := ioutil.ReadFile(common.Stage1ManifestPath(cdir))
	if err != nil {
		return -1, errwrap.Wrap(errors.New("error reading pod manifest"), err)
	}

	s1m := schema.ImageManifest{}
	if err := json.Unmarshal(b, &s1m); err != nil {
		return -1, errwrap.Wrap(errors.New("error unmarshaling stage1 manifest"), err)
	}

	if iv, ok := s1m.Annotations.Get(interfaceVersion); ok {
		v, err := strconv.Atoi(iv)
		if err != nil {
			return -1, errwrap.Wrap(errors.New("error parsing interface version"), err)
		}
		return v, nil
	}

	// "interface-version" annotation not found, assume version 1
	return 1, nil
}

func interfaceVersionSupportsHostname(version int) bool {
	return version > 1
}
