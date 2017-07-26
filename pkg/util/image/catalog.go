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

package image

import (
	"reflect"
)

type Catalog struct {
	Busybox string
	Ubuntu  string
}

func New() *Catalog {
	return &Catalog{
		Busybox: getBusyBoxImage(),
		Ubuntu:  getUbuntuImage(),
	}
}

func (c *Catalog) List() []string {
	v := reflect.ValueOf(c).Elem()
	values := make([]string, v.NumField())
	for i := 0; i < v.NumField(); i++ {
		values[i] = v.Field(i).Interface().(string)
	}

	return values
}
