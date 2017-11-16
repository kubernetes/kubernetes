/*
Copyright 2016 The Kubernetes Authors.

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

package types

import (
	"fmt"
	"regexp"
	"sort"
	"strings"

	"k8s.io/gengo/namer"
)

// ToGroupVersion turns "group/version" string into a GroupVersion struct. It reports error
// if it cannot parse the string.
func ToGroupVersion(gv string) (GroupVersion, error) {
	// this can be the internal version for the legacy kube types
	// TODO once we've cleared the last uses as strings, this special case should be removed.
	if (len(gv) == 0) || (gv == "/") {
		return GroupVersion{}, nil
	}

	switch strings.Count(gv, "/") {
	case 0:
		return GroupVersion{Group(gv), ""}, nil
	case 1:
		i := strings.Index(gv, "/")
		return GroupVersion{Group(gv[:i]), Version(gv[i+1:])}, nil
	default:
		return GroupVersion{}, fmt.Errorf("unexpected GroupVersion string: %v", gv)
	}
}

type sortableSliceOfVersions []string

func (a sortableSliceOfVersions) Len() int      { return len(a) }
func (a sortableSliceOfVersions) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a sortableSliceOfVersions) Less(i, j int) bool {
	vi, vj := strings.TrimLeft(a[i], "v"), strings.TrimLeft(a[j], "v")
	major := regexp.MustCompile("^[0-9]+")
	viMajor, vjMajor := major.FindString(vi), major.FindString(vj)
	viRemaining, vjRemaining := strings.TrimLeft(vi, viMajor), strings.TrimLeft(vj, vjMajor)
	switch {
	case len(viRemaining) == 0 && len(vjRemaining) == 0:
		return viMajor < vjMajor
	case len(viRemaining) == 0 && len(vjRemaining) != 0:
		// stable version is greater than unstable version
		return false
	case len(viRemaining) != 0 && len(vjRemaining) == 0:
		// stable version is greater than unstable version
		return true
	}
	// neither are stable versions
	if viMajor != vjMajor {
		return viMajor < vjMajor
	}
	// assuming at most we have one alpha or one beta version, so if vi contains "alpha", it's the lesser one.
	return strings.Contains(viRemaining, "alpha")
}

// Determine the default version among versions. If a user calls a group client
// without specifying the version (e.g., c.Core(), instead of c.CoreV1()), the
// default version will be returned.
func defaultVersion(versions []Version) Version {
	var versionStrings []string
	for _, version := range versions {
		versionStrings = append(versionStrings, string(version))
	}
	sort.Sort(sortableSliceOfVersions(versionStrings))
	return Version(versionStrings[len(versionStrings)-1])
}

// ToGroupVersionPackages is a helper function used by generators for groups.
func ToGroupVersionPackages(groups []GroupVersions, groupGoNames map[GroupVersion]string) []GroupVersionPackage {
	var groupVersionPackages []GroupVersionPackage
	for _, group := range groups {
		defaultVersion := defaultVersion(group.Versions)
		for _, version := range group.Versions {
			groupGoName := groupGoNames[GroupVersion{Group: group.Group, Version: version}]
			groupVersionPackages = append(groupVersionPackages, GroupVersionPackage{
				Group:                Group(namer.IC(group.Group.NonEmpty())),
				Version:              Version(namer.IC(version.String())),
				PackageAlias:         strings.ToLower(groupGoName + version.NonEmpty()),
				IsDefaultVersion:     version == defaultVersion && version != "",
				GroupGoName:          groupGoName,
				LowerCaseGroupGoName: namer.IL(groupGoName),
			})
		}
	}
	return groupVersionPackages
}

func ToGroupInstallPackages(groups []GroupVersions, groupGoNames map[GroupVersion]string) []GroupInstallPackage {
	var groupInstallPackages []GroupInstallPackage
	for _, group := range groups {
		defaultVersion := defaultVersion(group.Versions)
		groupGoName := groupGoNames[GroupVersion{Group: group.Group, Version: defaultVersion}]
		groupInstallPackages = append(groupInstallPackages, GroupInstallPackage{
			Group:               Group(namer.IC(group.Group.NonEmpty())),
			InstallPackageAlias: strings.ToLower(groupGoName),
		})
	}
	return groupInstallPackages
}

// NormalizeGroupVersion calls normalizes the GroupVersion.
//func NormalizeGroupVersion(gv GroupVersion) GroupVersion {
//	return GroupVersion{Group: gv.Group.NonEmpty(), Version: gv.Version, NonEmptyVersion: normalization.Version(gv.Version)}
//}
