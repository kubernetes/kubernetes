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

type GroupVersion struct {
	Group   string
	Version string
}

type GroupVersions struct {
	Group    string
	Versions []string
}

// GroupVersionPackage contains group name, version name, and the package name client-gen will generate for this group version.
type GroupVersionPackage struct {
	Group   string
	Version string
	// If a user calls a group client without specifying the version (e.g.,
	// c.Core(), instead of c.CoreV1()), the default version will be returned.
	IsDefaultVersion bool
	GroupVersion     string
	PackageName      string
}
