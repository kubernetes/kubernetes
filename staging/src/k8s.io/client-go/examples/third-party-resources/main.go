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
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/pkg/api"
	"k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/pkg/apis/extensions/v1beta1"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
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

	kind_created_here := false

	// initialize third party resource if it does not exist
	tpr, err := clientset.ExtensionsV1beta1().ThirdPartyResources().Get("example.k8s.io", metav1.GetOptions{})
	if err != nil {
		if errors.IsNotFound(err) {
			tpr := &v1beta1.ThirdPartyResource{
				ObjectMeta: metav1.ObjectMeta{
					Name: "example.k8s.io",
				},
				Versions: []v1beta1.APIVersion{
					{Name: "v1"},
				},
				Description: "An Example ThirdPartyResource",
			}

			result, err := clientset.ExtensionsV1beta1().ThirdPartyResources().Create(tpr)
			if err != nil {
				panic(err)
			}
			kind_created_here = true
			fmt.Printf("CREATED: %#v\nFROM: %#v\n", result, tpr)
		} else {
			panic(err)
		}
	} else {
		fmt.Printf("SKIPPING: already exists %#v\n", tpr)
	}

	// make a new config for our extension's API group, using the first config as a baseline
	var tprconfig *rest.Config
	tprconfig = config
	configureClient(tprconfig)

	tprclient, err := rest.RESTClientFor(tprconfig)
	if err != nil {
		panic(err)
	}

	var example Example

	err = tprclient.Get().
		Resource("examples").
		Namespace(v1.NamespaceDefault).
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
			req := tprclient.Post().
				Resource("examples").
				Namespace(v1.NamespaceDefault).
				Body(example)
			err = req.
				Do().Into(&result)

			if err != nil {
				fmt.Println()
				if kind_created_here {
					glog.Infoln("Probably because of delay issue noted in https://github.com/kubernetes/features/issues/95 ...")
				}
				glog.Fatalf("Unable to create example1 --- request=%#v, result=%#v, err=%#v", req, result, err)
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
	err = tprclient.Get().Resource("examples").Do().Into(&exampleList)
	if err != nil {
		panic(err)
	}
	fmt.Printf("LIST: %#v\n", exampleList)

	fmt.Printf("Starting watch!\n")
	watch(tprclient)
}

func buildConfig(kubeconfig string) (*rest.Config, error) {
	if kubeconfig != "" {
		return clientcmd.BuildConfigFromFlags("", kubeconfig)
	}
	return rest.InClusterConfig()
}

func configureClient(config *rest.Config) {
	groupversion := schema.GroupVersion{
		Group:   "k8s.io",
		Version: "v1",
	}

	config.GroupVersion = &groupversion
	config.APIPath = "/apis"
	config.ContentType = runtime.ContentTypeJSON
	config.NegotiatedSerializer = serializer.DirectCodecFactory{CodecFactory: scheme.Codecs}

	schemeBuilder := runtime.NewSchemeBuilder(
		func(scheme *runtime.Scheme) error {
			scheme.AddKnownTypes(
				groupversion,
				&Example{},
				&ExampleList{},
			)
			return nil
		})
	metav1.AddToGroupVersion(scheme.Scheme, groupversion)
	schemeBuilder.AddToScheme(scheme.Scheme)
}

func watch(client *rest.RESTClient) {

	stop := make(chan struct{}, 1)
	source := cache.NewListWatchFromClient(
		client,
		"examples",
		api.NamespaceAll,
		fields.Everything())

	store, controller := cache.NewInformer(
		source,

		// The object type.
		&Example{},

		// resyncPeriod
		// Every resyncPeriod, all resources in the cache will retrigger events.
		// Set to 0 to disable the resync.
		time.Second*60,

		// Your custom resource event handlers.
		cache.ResourceEventHandlerFuncs{
			// Takes a single argument of type interface{}.
			// Called on controller startup and when new resources are created.
			AddFunc: create,

			// Takes two arguments of type interface{}.
			// Called on resource update and every resyncPeriod on existing resources.
			UpdateFunc: update,

			// Takes a single argument of type interface{}.
			// Called on resource deletion.
			DeleteFunc: delete,
		})

	// store can be used to List and Get
	// NEVER modify objects from the store. It's a read-only, local cache.
	fmt.Println("listing examples from store:")
	for _, obj := range store.List() {
		example := obj.(*Example)

		// This will likely be empty the first run, but may not
		fmt.Printf("%#v\n", example)
	}

	// the controller run starts the event processing loop
	go controller.Run(stop)

	// and now we block on a signal
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)
	s := <-signals
	fmt.Printf("received signal %#v, exiting...\n", s)
	close(stop)
	os.Exit(0)
}

// Handler functions as per the controller above.
// Note the coercion of the interface{} into a pointer of the expected type.

func create(obj interface{}) {
	example := obj.(*Example)

	fmt.Println("CREATED:", printExample(example))
}

func update(old, new interface{}) {
	oldExample := old.(*Example)
	newExample := new.(*Example)

	fmt.Printf("UPDATED:\n  old: %s\n  new: %s\n", printExample(oldExample), printExample(newExample))
}

func delete(obj interface{}) {
	example := obj.(*Example)

	fmt.Println("DELETED:", printExample(example))
}

// convenience functions
func printExample(example *Example) string {
	return fmt.Sprintf("%s/%s, APIVersion: %s, Kind: %s, Value: %#v", example.Metadata.Namespace, example.Metadata.Name, example.APIVersion, example.Kind, example)
}
