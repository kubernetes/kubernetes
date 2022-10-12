/*
Copyright 2017 The Kubernetes Authors.

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

package cached

import (
	"sync"

	"k8s.io/client-go/openapi"
)

type groupversion struct {
	delegate openapi.GroupVersion

	jsonOnce  sync.Once
	jsonBytes []byte
	jsonErr   error

	pbOnce  sync.Once
	pbBytes []byte
	pbErr   error
}

func newGroupVersion(delegate openapi.GroupVersion) *groupversion {
	return &groupversion{
		delegate: delegate,
	}
}

func (g *groupversion) SchemaPB() ([]byte, error) {
	g.pbOnce.Do(func() {
		g.pbBytes, g.pbErr = g.delegate.SchemaPB()
	})

	return g.pbBytes, g.pbErr
}

func (g *groupversion) SchemaJSON() ([]byte, error) {
	g.jsonOnce.Do(func() {
		g.jsonBytes, g.jsonErr = g.delegate.SchemaJSON()
	})

	return g.jsonBytes, g.jsonErr
}
