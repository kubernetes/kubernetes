/*
Copyright 2021 The Kubernetes Authors.

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

package util

import (
	"os"
)

// EnvDefault returns either the provided environment variable for the given
// `key` or the default value `def` if not set.
func EnvDefault(key, def string) string {
	value, ok := os.LookupEnv(key)
	if !ok || value == "" {
		return def
	}
	return value
}

// IsEnvSet returns true if an environment variable is set
func IsEnvSet(key string) bool {
	_, ok := os.LookupEnv(key)
	return ok
}
