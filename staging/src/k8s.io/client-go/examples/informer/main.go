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

// Note: the example only works with the code within the same release/branch.
package main

import (
	"flag"
	"time"

	"github.com/golang/glog"

	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/clientcmd"
	// Only required to authenticate against GKE clusters
	// _ "k8s.io/client-go/plugin/pkg/client/auth/gcp"
)

func main() {
	var kubeconfig string
	var master string

	flag.StringVar(&kubeconfig, "kubeconfig", "", "absolute path to the kubeconfig file")
	flag.StringVar(&master, "master", "", "master url")
	flag.Parse()

	// BuildConfigFromFlags builds configs from a master url or a kubeconfig
	// filepath. If neither masterUrl nor kubeconfigPath are passed, it falls back
	// to inClusterConfig. If inClusterConfig fails, it falls back to the default
	// config.
	config, err := clientcmd.BuildConfigFromFlags(master, kubeconfig)
	if err != nil {
		glog.Fatal(err)
	}

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		glog.Fatal(err)
	}

	factory := informers.NewSharedInformerFactory(clientset, time.Second*10)

	informer := factory.Core().V1().Pods().Informer()

	stop := make(chan struct{})
	defer close(stop)

	// the controller run starts the event processing loop
	go informer.Run(stop)

	// wait until a store finished its initial synchronization
	cache.WaitForCacheSync(stop, informer.HasSynced)

	informer.AddEventHandler(
		// Your custom resource event handlers.
		cache.ResourceEventHandlerFuncs{
			// Called on creation
			AddFunc: func(obj interface{}) {
				pod := obj.(*v1.Pod)
				glog.Infof("POD CREATED: %s/%s", pod.Namespace, pod.Name)
			},
			// Called on resource update and every resyncPeriod on existing resources.
			// The Generation will not change unless the Pod has changed.
			UpdateFunc: func(old, new interface{}) {
				oldPod := old.(*v1.Pod)
				newPod := new.(*v1.Pod)
				glog.Infof(
					"POD UPDATED. %s/%s %s",
					oldPod.Namespace, oldPod.Name, newPod.Status.Phase,
				)
			},
			// Called on resource deletion.
			DeleteFunc: func(obj interface{}) {
				pod := obj.(*v1.Pod)
				glog.Infof("POD DELETED: %s/%s", pod.Namespace, pod.Name)
			},
		},
	)

	select {}
}
