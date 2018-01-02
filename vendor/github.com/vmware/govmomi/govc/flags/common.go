/*
Copyright (c) 2015-2016 VMware, Inc. All Rights Reserved.

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
package flags

import "sync"

// Key type for storing flag instances in a context.Context.
type flagKey string

// Type to help flags out with only registering/processing once.
type common struct {
	register sync.Once
	process  sync.Once
}

func (c *common) RegisterOnce(fn func()) {
	c.register.Do(fn)
}

func (c *common) ProcessOnce(fn func() error) (err error) {
	c.process.Do(func() {
		err = fn()
	})
	return err
}
