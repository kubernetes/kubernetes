/*
Copyright 2014 The Kubernetes Authors.

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

package framework

import (
	"fmt"
	"math/rand"

	"k8s.io/kubernetes/pkg/storage"

	consulapi "github.com/hashicorp/consul/api"
)

var consulClient *consulapi.Client

func init() {
	var err error
	consulClient, err = consulapi.NewClient(consulapi.DefaultConfig())
	if err != nil {
		panic(fmt.Sprintf("Failed to instantiate consulClient: %s", err))
	}
}

func WithConsulKey(store storage.Interface, f func(string)) {
	prefix := fmt.Sprintf("/test-%d", rand.Int63())
	defer consulClient.KV().Delete(prefix, nil)
	f(prefix)
}
