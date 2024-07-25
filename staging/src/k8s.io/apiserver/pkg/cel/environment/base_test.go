/*
Copyright 2023 The Kubernetes Authors.

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

package environment

import (
	"sort"
	"strings"
	"testing"

	"github.com/google/cel-go/cel"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/cel/library"
)

// BenchmarkLoadBaseEnv is expected to be very fast, because a
// a cached environment is loaded for each MustBaseEnvSet call.
func BenchmarkLoadBaseEnv(b *testing.B) {
	ver := DefaultCompatibilityVersion()
	MustBaseEnvSet(ver, true)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MustBaseEnvSet(ver, true)
	}
}

// BenchmarkLoadBaseEnvDifferentVersions is expected to be relatively slow, because a
// a new environment must be created for each MustBaseEnvSet call.
func BenchmarkLoadBaseEnvDifferentVersions(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MustBaseEnvSet(version.MajorMinor(1, uint(i)), true)
	}
}

// TestLibraryCoverage lints the management of libraries in baseOpts by
// checking for:
//
//   - No gaps and overlap in library inclusion, including when libraries are version bumped
//   - RemovedVersion is always greater than IntroducedVersion
//   - Libraries are not removed once added (although they can be replaced with new versions)
func TestLibraryCoverage(t *testing.T) {
	vops := make([]VersionedOptions, len(baseOpts))
	copy(vops, baseOpts)
	sort.SliceStable(vops, func(i, j int) bool {
		return vops[i].IntroducedVersion.LessThan(vops[j].IntroducedVersion)
	})

	tracked := map[string]*versionTracker{}

	for _, vop := range vops {
		if vop.RemovedVersion != nil {
			if vop.IntroducedVersion == nil {
				t.Errorf("VersionedOptions with RemovedVersion %v is missing required IntroducedVersion", vop.RemovedVersion)
			}
			if !vop.IntroducedVersion.LessThan(vop.RemovedVersion) {
				t.Errorf("VersionedOptions with IntroducedVersion %s must be less than RemovedVersion %v", vop.IntroducedVersion, vop.RemovedVersion)
			}
		}

		for _, name := range librariesInVersions(t, vop) {
			versionTracking, ok := tracked[name]
			if !ok {
				versionTracking = &versionTracker{}
				tracked[name] = versionTracking
			}
			if versionTracking.added != nil {
				t.Errorf("Did not expect %s library to be added again at version %v. It was already added at version %v", name, vop.IntroducedVersion, versionTracking.added)
			} else {
				versionTracking.added = vop.IntroducedVersion
				if versionTracking.removed != nil {
					if versionTracking.removed.LessThan(vop.IntroducedVersion) {
						t.Errorf("Did not expect gap in presence of %s library. It was "+
							"removed in %v and not added again until %v. When versioning "+
							"libraries, introduce a new version of the library as the same "+
							"kubernetes version that the old version of the library is removed.", name, versionTracking.removed, vop.IntroducedVersion)
					} else if vop.IntroducedVersion.LessThan(versionTracking.removed) {
						t.Errorf("Did not expect overlap in presence of %s library. It was "+
							"added again at version %v while scheduled to be removed at %v. When versioning "+
							"libraries, introduce a new version of the library as the same "+
							"kubernetes version that the old version of the library is removed.", name, vop.IntroducedVersion, versionTracking.removed)
					}
				}
				versionTracking.removed = nil
			}
			if vop.RemovedVersion != nil {
				if versionTracking.removed != nil {
					t.Errorf("Unexpected RemovedVersion of %v for library %s already removed at version %v", vop.RemovedVersion, name, versionTracking.removed)
				}
				versionTracking.added = nil
				versionTracking.removed = vop.RemovedVersion
			}
		}
	}
	for name, lib := range tracked {
		if lib.removed != nil {
			t.Errorf("Unexpected RemovedVersion of %v for library %s without replacement. "+
				"For backward compatibility, libraries should not be removed without being replaced by a new version.", lib.removed, name)
		}
	}
}

func TestKnownLibraries(t *testing.T) {
	known := sets.New[string]()
	used := sets.New[string]()

	for _, lib := range library.KnownLibraries() {
		known.Insert(lib.LibraryName())
	}
	for _, libName := range MustBaseEnvSet(version.MajorMinor(1, 0), true).storedExpressions.Libraries() {
		if strings.HasPrefix(libName, "cel.lib") { // ignore core libs
			continue
		}
		used.Insert(libName)
	}

	unexpected := used.Difference(known)

	if len(unexpected) != 0 {
		t.Errorf("Expected all libraries in the base environment to be included k8s.io/apiserver/pkg/cel/library's KnownLibraries, but found missing libraries: %v", unexpected)
	}

}

func librariesInVersions(t *testing.T, vops ...VersionedOptions) []string {
	env, err := cel.NewCustomEnv()
	if err != nil {
		t.Fatalf("Error creating env: %v", err)
	}
	for _, vop := range vops {
		env, err = env.Extend(vop.EnvOptions...)
		if err != nil {
			t.Fatalf("Error updating env: %v", err)
		}
	}
	libs := env.Libraries()
	return libs
}

type versionTracker struct {
	added   *version.Version
	removed *version.Version
}
