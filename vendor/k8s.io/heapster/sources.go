// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"fmt"
	"strings"

	"k8s.io/heapster/extpoints"
	"k8s.io/heapster/sinks/cache"
	"k8s.io/heapster/sources/api"
)

func newSources(c cache.Cache) ([]api.Source, error) {
	var sources []api.Source
	var errors []string
	for _, u := range argSources {
		factory := extpoints.SourceFactories.Lookup(u.Key)
		if factory == nil {
			return nil, fmt.Errorf("Unknown source: %s", u.Key)
		}

		createdSources, err := factory(&u.Val, c)
		if err != nil {
			errors = append(errors, err.Error())
		}
		sources = append(sources, createdSources...)
	}
	var err error
	if len(errors) > 0 {
		err = fmt.Errorf("encountered following errors while setting up sources - %v", strings.Join(errors, "; "))
	}

	return sources, err
}
