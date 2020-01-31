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

// Version is string.
type Version string

func (v Version) String() string {
	return string(v)
}

// NonEmpty retunrs version as string.
func (v Version) NonEmpty() string {
	if v == "" {
		return "internalVersion"
	}
	return v.String()
}

// Group is string.
type Group string

func (g Group) String() string {
	return string(g)
}

// NonEmpty retunrs group as string.
func (g Group) NonEmpty() string {
	if g == "api" {
		return "core"
	}
	return string(g)
}

// PackageVersion contains package and it's version..
type PackageVersion struct {
	Version
	// The fully qualified package, e.g. k8s.io/kubernetes/pkg/apis/apps, where the types.go is found.
	Package string
}

// GroupVersion contains version for this group.
type GroupVersion struct {
	Group   Group
	Version Version
}

// GroupVersions contains package name and versions for this group.
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

// GroupInstallPackage contains group and install-package alias for this group.
type GroupInstallPackage struct {
	Group               Group
	InstallPackageAlias string
}
