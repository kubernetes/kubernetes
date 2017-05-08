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
	"context"
	"flag"
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/pkg/api"
	"k8s.io/client-go/pkg/apis/extensions/v1beta1"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	// Uncomment the following line to load the gcp plugin (only required to authenticate against GKE clusters).
	// _ "k8s.io/client-go/plugin/pkg/client/auth/gcp"
)

func main() {
	kubeconfig := flag.String("kubeconfig", "", "Path to a kube config. Only required if out-of-cluster.")
	flag.Parse()

	// Create the client config. Use kubeconfig if given, otherwise assume in-cluster.
	config, err := buildConfig(*kubeconfig)
	if err != nil {
		panic(err)
	}

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		panic(err)
	}

	// initialize third party resource if it does not exist
	tpr, err := clientset.ExtensionsV1beta1().ThirdPartyResources().Get(ExampleResourceName, metav1.GetOptions{})
	if err != nil {
		if errors.IsNotFound(err) {
			tpr := &v1beta1.ThirdPartyResource{
				ObjectMeta: metav1.ObjectMeta{
					Name: ExampleResourceName,
				},
				Versions: []v1beta1.APIVersion{
					{Name: ExampleResourceVersion},
				},
				Description: ExampleResourceDescription,
			}

			result, err := clientset.ExtensionsV1beta1().ThirdPartyResources().Create(tpr)
			if err != nil {
				panic(err)
			}
			fmt.Printf("CREATED: %#v\nFROM: %#v\n", result, tpr)

			// See the issue https://github.com/kubernetes/features/issues/95
			// ("Make new TPRs available immediately after the create request succeeds")
			fmt.Print("Sleeping to make sure the TPR is processed and available")
			time.Sleep(5 * time.Second)
		} else {
			panic(err)
		}
	} else {
		fmt.Printf("SKIPPING: already exists %#v\n", tpr)
	}

	// make a new config for our extension's API group, using the first config as a baseline
	exampleClient, exampleScheme, err := NewClient(config)
	if err != nil {
		panic(err)
	}

	// start a watcher on instances of our TPR
	watcher := Watcher{
		clientset:     clientset,
		exampleClient: exampleClient,
		exampleScheme: exampleScheme,
	}

	ctx, cancelFunc := context.WithCancel(context.Background())
	defer cancelFunc()
	go watcher.Run(ctx)

	// The sleep below is just to make sure that the watcher.Run() goroutine has successfully executed
	// and the watcher is handling the events about Example TPR instances.
	// In the normal application there is no need for it, because:
	// 1. It's unlikely to create a watcher and a TPR instance at the same time in the same application.
	// 2. The application with watcher would most probably keep running instead of exiting right after the watcher startup.
	time.Sleep(5 * time.Second)

	// GET/POST an instance of our TPR
	var example Example

	err = exampleClient.Get().
		Resource(ExampleResourcePath).
		Namespace(api.NamespaceDefault).
		Name("example1").
		Do().Into(&example)

	if err != nil {
		if errors.IsNotFound(err) {
			// Create an instance of our TPR
			example := &Example{
				Metadata: metav1.ObjectMeta{
					Name: "example1",
				},
				Spec: ExampleSpec{
					Foo: "hello",
					Bar: true,
				},
			}

			var result Example
			err = exampleClient.Post().
				Resource(ExampleResourcePath).
				Namespace(api.NamespaceDefault).
				Body(example).
				Do().Into(&result)

			if err != nil {
				panic(err)
			}
			fmt.Printf("CREATED: %#v\n", result)
		} else {
			panic(err)
		}
	} else {
		fmt.Printf("GET: %#v\n", example)
	}

	// Fetch a list of our TPRs
	exampleList := ExampleList{}
	err = exampleClient.Get().Resource(ExampleResourcePath).Do().Into(&exampleList)
	if err != nil {
		panic(err)
	}
	fmt.Printf("LIST: %#v\n", exampleList)
}

func buildConfig(kubeconfig string) (*rest.Config, error) {
	if kubeconfig != "" {
		return clientcmd.BuildConfigFromFlags("", kubeconfig)
	}
	return rest.InClusterConfig()
}
