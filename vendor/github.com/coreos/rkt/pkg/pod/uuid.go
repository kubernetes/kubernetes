// Copyright 2015 The rkt Authors
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

package pod

import (
	"bytes"
	"errors"
	"fmt"
	"io/ioutil"
	"strings"

	"github.com/appc/spec/schema/types"
	"github.com/hashicorp/errwrap"
)

// matchUUID attempts to match the uuid specified as uuid against all pods present.
// An array of matches is returned, which may be empty when nothing matches.
func matchUUID(dataDir, uuid string) ([]string, error) {
	if uuid == "" {
		return nil, types.ErrNoEmptyUUID
	}

	ls, err := listPods(dataDir, IncludeMostDirs)
	if err != nil {
		return nil, err
	}

	var matches []string
	for _, p := range ls {
		if strings.HasPrefix(p, uuid) {
			matches = append(matches, p)
		}
	}

	return matches, nil
}

// resolveUUID attempts to resolve the uuid specified as uuid against all pods present.
// An unambiguously matched uuid or nil is returned.
func resolveUUID(dataDir, uuid string) (*types.UUID, error) {
	uuid = strings.ToLower(uuid)
	m, err := matchUUID(dataDir, uuid)
	if err != nil {
		return nil, err
	}

	if len(m) == 0 {
		return nil, fmt.Errorf("no matches found for %q", uuid)
	}

	if len(m) > 1 {
		return nil, fmt.Errorf("ambiguous uuid, %d matches", len(m))
	}

	u, err := types.NewUUID(m[0])
	if err != nil {
		return nil, errwrap.Wrap(errors.New("invalid UUID"), err)
	}

	return u, nil
}

// ReadUUIDFromFile reads the uuid string from the given path.
func ReadUUIDFromFile(path string) (string, error) {
	uuid, err := ioutil.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(bytes.TrimSpace(uuid)), nil
}

// WriteUUIDToFile writes the uuid string to the given path.
func WriteUUIDToFile(uuid *types.UUID, path string) error {
	return ioutil.WriteFile(path, []byte(uuid.String()), 0644)
}
