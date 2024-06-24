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

import "strings"

type Version string

func (v Version) String() string {
	return string(v)
}

func (v Version) NonEmpty() string {
	if v == "" {
		return "internalVersion"
	}
	return v.String()
}

func (v Version) PackageName() string {
	return strings.ToLower(v.NonEmpty())
}

type Group string

func (g Group) String() string {
	return string(g)
}

func (g Group) NonEmpty() string {
	if g == "api" {
		return "core"
	}
	return string(g)
}

func (g Group) PackageName() string {
	parts := strings.Split(g.NonEmpty(), ".")
	if parts[0] == "internal" && len(parts) > 1 {
		return strings.ToLower(parts[1] + parts[0])
	}
	return strings.ToLower(parts[0])
}

type Kind string

type PackageVersion struct {
	Version
	// The fully qualified package, e.g. k8s.io/kubernetes/pkg/apis/apps, where the types.go is found.
	Package string
}

type GroupVersion struct {
	Group   Group
	Version Version
}

type GroupVersionKind struct {
	Group   Group
	Version Version
	Kind    Kind
}

func (gv GroupVersion) ToAPIVersion() string {
	if len(gv.Group) > 0 && gv.Group.NonEmpty() != "core" {
		return gv.Group.String() + "/" + gv.Version.String()
	} else {
		return gv.Version.String()
	}
}

func (gv GroupVersion) WithKind(kind Kind) GroupVersionKind {
	return GroupVersionKind{Group: gv.Group, Version: gv.Version, Kind: kind}
}

type GroupVersions struct {
	// The name of the package for this group, e.g. apps.
	PackageName string
	Group       Group
	Versions    []PackageVersion
}

// GroupVersionInfo contains all the info around a group version.
type GroupVersionInfo struct {
	Group                Group
	Version              Version
	PackageAlias         string
	GroupGoName          string
	LowerCaseGroupGoName string
}

type GroupInstallPackage struct {
	Group               Group
	InstallPackageAlias string
}
