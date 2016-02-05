/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package main

import (
	"flag"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
)

var (
	resyncPeriod = 30 * time.Minute
	config       *client.Config

	kube                      *client.Client
	endpoints, services, pods cache.Store
)

func init() {
	flag.DurationVar(&resyncPeriod, "resync-period", resyncPeriod, "The delay between forced syncs")

	// Use in-cluster config or provide options
	var err error
	config, err = client.InClusterConfig()
	if err != nil {
		config = &client.Config{}
		if err = client.SetKubernetesDefaults(config); err != nil {
			panic(err)
		}
	}
	flag.StringVar(&config.Host, "master", config.Host, "The address of the Kubernetes API server")
}

func main() {
	flag.Parse()

	kube = client.NewOrDie(config)

	syncChan := make(chan bool, 1)

	watches := []struct {
		name     string
		storeRef *cache.Store
		instance interface{}
	}{
		{"endpoints", &endpoints, &api.Endpoints{}},
		{"services", &services, &api.Service{}},
		{"pods", &pods, &api.Pod{}},
	}
	for _, w := range watches {
		lw := cache.NewListWatchFromClient(kube, w.name, api.NamespaceAll, fields.Everything())
		*w.storeRef = cache.NewUndeltaStore(func([]interface{}) { syncChan <- true }, cache.MetaNamespaceKeyFunc)
		cache.NewReflector(lw, w.instance, *w.storeRef, resyncPeriod).Run()
	}

	for range syncChan {
		sync()
	}
}
