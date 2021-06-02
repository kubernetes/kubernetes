/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package path

import (
	"errors"
	"os"
)

// LinkTreatment is the base type for constants used by Exists that indicate
// how symlinks are treated for existence checks.
type LinkTreatment int

const (
	// CheckFollowSymlink follows the symlink and verifies that the target of
	// the symlink exists.
	CheckFollowSymlink LinkTreatment = iota

	// CheckSymlinkOnly does not follow the symlink and verifies only that they
	// symlink itself exists.
	CheckSymlinkOnly
)

// ErrInvalidLinkTreatment indicates that the link treatment behavior requested
// is not a valid behavior.
var ErrInvalidLinkTreatment = errors.New("unknown link behavior")

// Exists checks if specified file, directory, or symlink exists. The behavior
// of the test depends on the linkBehaviour argument. See LinkTreatment for
// more details.
func Exists(linkBehavior LinkTreatment, filename string) (bool, error) {
	var err error

	if linkBehavior == CheckFollowSymlink {
		_, err = os.Stat(filename)
	} else if linkBehavior == CheckSymlinkOnly {
		_, err = os.Lstat(filename)
	} else {
		return false, ErrInvalidLinkTreatment
	}

	if os.IsNotExist(err) {
		return false, nil
	} else if err != nil {
		return false, err
	}
	return true, nil
}

// ReadDirNoStat returns a string of files/directories contained
// in dirname without calling lstat on them.
func ReadDirNoStat(dirname string) ([]string, error) {
	if dirname == "" {
		dirname = "."
	}

	f, err := os.Open(dirname)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return f.Readdirnames(-1)
}
