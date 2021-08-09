/*
Copyright 2016 The Kubernetes Authors.

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
	"fmt"
	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/tools/cache"
	"log"
	"net/http"
	"path/filepath"
	"time"

	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
	//
	// Uncomment to load all auth plugins
	// _ "k8s.io/client-go/plugin/pkg/client/auth"
	//
	// Or uncomment to load specific auth plugins
	// _ "k8s.io/client-go/plugin/pkg/client/auth/azure"
	// _ "k8s.io/client-go/plugin/pkg/client/auth/gcp"
	// _ "k8s.io/client-go/plugin/pkg/client/auth/oidc"
	// _ "k8s.io/client-go/plugin/pkg/client/auth/openstack"
)

func main() {
	var kubeconfig *string
	if home := homedir.HomeDir(); home != "" {
		kubeconfig = flag.String("kubeconfig", filepath.Join(home, ".kube", "config"), "(optional) absolute path to the kubeconfig file")
	} else {
		kubeconfig = flag.String("kubeconfig", "", "absolute path to the kubeconfig file")
	}
	flag.Parse()

	// use the current context in kubeconfig
	config, err := clientcmd.BuildConfigFromFlags("", *kubeconfig)
	if err != nil {
		panic(err.Error())
	}

	// create the clientset
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		panic(err.Error())
	}

	sharedOptions := []informers.SharedInformerOption{
		// filter namespace
		informers.WithNamespace(v1.NamespaceAll),
	}
	infos := informers.NewSharedInformerFactoryWithOptions(clientset, time.Second*30, sharedOptions...)

	pod := infos.Core().V1().Pods().Informer()
	pod.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			p, _ := obj.(*v1.Pod)
			fmt.Printf("%v.%v has created\n", p.Name, p.Namespace)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			op, _ := oldObj.(*v1.Pod)
			np, _ := newObj.(*v1.Pod)
			fmt.Printf("%v.%v has updated to %v.%v\n", op.Name, op.Namespace, np.Name, np.Namespace)
		},
		DeleteFunc: func(obj interface{}) {
			p, _ := obj.(*v1.Pod)
			fmt.Printf("%v.%v has deleted\n", p.Name, p.Namespace)
		},
	})

	stop := make(chan struct{})
	go pod.Run(stop)

	// keep alive
	log.Fatal(http.ListenAndServe(":8080", nil))
}
