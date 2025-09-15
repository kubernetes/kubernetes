/*
Copyright 2015 The Kubernetes Authors.

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

package testing

import (
	"os"
	"path"
)

// PathPrefix returns the prefix set via the ETCD_PREFIX environment variable (if any).
func PathPrefix() string {
	pref := os.Getenv("ETCD_PREFIX")
	if pref == "" {
		pref = "registry"
	}
	return path.Join("/", pref)
}

// AddPrefix adds the ETCD_PREFIX to the provided key
func AddPrefix(in string) string {
	return path.Join(PathPrefix(), in)
}
