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
	"net/url"

	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/util/env"

	"github.com/golang/glog"
	consulapi "github.com/hashicorp/consul/api"
)

func GetConsulURLFromEnv() string {
	url := env.GetEnvAsStringOrFallback("KUBE_INTEGRATION_CONSUL_URL", "http://127.0.0.1:8500")
	glog.V(4).Infof("Using KUBE_INTEGRATION_CONSUL_URL=%q", url)
	return url
}

func NewConsulClient() *consulapi.Client {
	config := consulapi.DefaultConfig()

	parsedUrl, err := url.Parse(GetConsulURLFromEnv())
	if err != nil {
		glog.Fatalf("Failed to parse KUBE_INTEGRATION_CONSUL_URL: %s", err)
	}

	config.Address = parsedUrl.Host
	config.Scheme = parsedUrl.Scheme

	consulClient, err := consulapi.NewClient(config)
	if err != nil {
		glog.Fatalf("Failed to instantiate consulClient: %s", err)
	}

	return consulClient
}

func WithConsulKey(store storage.Interface, f func(string)) {
	prefix := fmt.Sprintf("/test-%d", rand.Int63())
	defer NewConsulClient().KV().Delete(prefix, nil)
	f(prefix)
}
