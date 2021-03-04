/*
Copyright 2020 The Kubernetes Authors.

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

package deprecation

import (
	"fmt"
	"regexp"
	"strconv"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/version"
)

type apiLifecycleDeprecated interface {
	APILifecycleDeprecated() (major, minor int)
}

type apiLifecycleRemoved interface {
	APILifecycleRemoved() (major, minor int)
}

type apiLifecycleReplacement interface {
	APILifecycleReplacement() schema.GroupVersionKind
}

// extract all digits at the beginning of the string
var leadingDigits = regexp.MustCompile(`^(\d+)`)

// MajorMinor parses a numeric major/minor version from the provided version info.
// The minor version drops all characters after the first non-digit character:
//   version.Info{Major:"1", Minor:"2+"} -> 1,2
//   version.Info{Major:"1", Minor:"2.3-build4"} -> 1,2
func MajorMinor(v version.Info) (int, int, error) {
	major, err := strconv.Atoi(v.Major)
	if err != nil {
		return 0, 0, err
	}
	minor, err := strconv.Atoi(leadingDigits.FindString(v.Minor))
	if err != nil {
		return 0, 0, err
	}
	return major, minor, nil
}

// IsDeprecated returns true if obj implements APILifecycleDeprecated() and returns
// a major/minor version that is non-zero and is <= the specified current major/minor version.
func IsDeprecated(obj runtime.Object, currentMajor, currentMinor int) bool {
	deprecated, isDeprecated := obj.(apiLifecycleDeprecated)
	if !isDeprecated {
		return false
	}

	deprecatedMajor, deprecatedMinor := deprecated.APILifecycleDeprecated()
	// no deprecation version expressed
	if deprecatedMajor == 0 && deprecatedMinor == 0 {
		return false
	}
	// no current version info available
	if currentMajor == 0 && currentMinor == 0 {
		return true
	}
	// compare deprecation version to current version
	if deprecatedMajor > currentMajor {
		return false
	}
	if deprecatedMajor == currentMajor && deprecatedMinor > currentMinor {
		return false
	}
	return true
}

// RemovedRelease returns the major/minor version in which the given object is unavailable (in the form "<major>.<minor>")
// if the object implements APILifecycleRemoved() to indicate a non-zero removal version, and returns an empty string otherwise.
func RemovedRelease(obj runtime.Object) string {
	if removed, hasRemovalInfo := obj.(apiLifecycleRemoved); hasRemovalInfo {
		removedMajor, removedMinor := removed.APILifecycleRemoved()
		if removedMajor != 0 || removedMinor != 0 {
			return fmt.Sprintf("%d.%d", removedMajor, removedMinor)
		}
	}
	return ""
}

// WarningMessage returns a human-readable deprecation warning if the object implements APILifecycleDeprecated()
// to indicate a non-zero deprecated major/minor version and has a populated GetObjectKind().GroupVersionKind().
func WarningMessage(obj runtime.Object) string {
	deprecated, isDeprecated := obj.(apiLifecycleDeprecated)
	if !isDeprecated {
		return ""
	}

	deprecatedMajor, deprecatedMinor := deprecated.APILifecycleDeprecated()
	if deprecatedMajor == 0 && deprecatedMinor == 0 {
		return ""
	}

	gvk := obj.GetObjectKind().GroupVersionKind()
	if gvk.Empty() {
		return ""
	}
	deprecationWarning := fmt.Sprintf("%s %s is deprecated in v%d.%d+", gvk.GroupVersion().String(), gvk.Kind, deprecatedMajor, deprecatedMinor)

	if removed, hasRemovalInfo := obj.(apiLifecycleRemoved); hasRemovalInfo {
		removedMajor, removedMinor := removed.APILifecycleRemoved()
		if removedMajor != 0 || removedMinor != 0 {
			deprecationWarning = deprecationWarning + fmt.Sprintf(", unavailable in v%d.%d+", removedMajor, removedMinor)
		}
	}

	if replaced, hasReplacement := obj.(apiLifecycleReplacement); hasReplacement {
		replacement := replaced.APILifecycleReplacement()
		if !replacement.Empty() {
			deprecationWarning = deprecationWarning + fmt.Sprintf("; use %s %s", replacement.GroupVersion().String(), replacement.Kind)
		}
	}

	return deprecationWarning
}
