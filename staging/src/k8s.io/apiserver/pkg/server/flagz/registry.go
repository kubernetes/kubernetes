/*
Copyright 2024 The Kubernetes Authors.

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

package flagz

import (
	"sync"
)

type registry struct {
	// reader is a Reader where we can get the flags.
	reader Reader
	// deprecatedVersionsMap is a map of deprecated flagz versions.
	deprecatedVersionsMap map[string]bool
	// cachedPlainTextResponse is a cached response of the flagz endpoint.
	cachedPlainTextResponse []byte
	// cachedPlainTextResponseLock is a lock for the cachedPlainTextResponse.
	cachedPlainTextResponseLock sync.Mutex
}

// Option is a function to configure registry.
type Option func(reg *registry)

func (r *registry) deprecatedVersions() map[string]bool {
	return r.deprecatedVersionsMap
}
