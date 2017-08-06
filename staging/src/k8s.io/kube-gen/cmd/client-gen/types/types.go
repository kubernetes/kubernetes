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

type GroupVersion struct {
	Group   Group
	Version Version
}

type GroupVersions struct {
	Group    Group
	Versions []Version
}

// GroupVersionPackage contains group name, version name, and the package name client-gen will generate for this group version.
type GroupVersionPackage struct {
	Group   Group
	Version Version
	// If a user calls a group client without specifying the version (e.g.,
	// c.Core(), instead of c.CoreV1()), the default version will be returned.
	IsDefaultVersion      bool
	GroupVersion          string
	LowerCaseGroupVersion string
	PackageName           string
}

type GroupInstallPackage struct {
	Group              Group
	InstallPackageName string
}
