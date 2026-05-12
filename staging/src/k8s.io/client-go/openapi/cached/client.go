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

type client struct {
	delegate openapi.Client

	once   sync.Once
	result map[string]openapi.GroupVersion
	err    error
}

func NewClient(other openapi.Client) openapi.Client {
	return &client{
		delegate: other,
	}
}

func (c *client) Paths() (map[string]openapi.GroupVersion, error) {
	c.once.Do(func() {
		uncached, err := c.delegate.Paths()
		if err != nil {
			c.err = err
			return
		}

		result := make(map[string]openapi.GroupVersion, len(uncached))
		for k, v := range uncached {
			result[k] = newGroupVersion(v)
		}
		c.result = result
	})
	return c.result, c.err
}
