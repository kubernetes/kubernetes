/*
Copyright 2018 The Kubernetes Authors.

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

//This is an example of polymorphic scale client.

package main

import (
	"flag"
	"os"

	"github.com/golang/glog"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/scale"
	"k8s.io/client-go/tools/clientcmd"
	clientset "k8s.io/sample-controller/pkg/client/clientset/versioned"
)

const (
	namespace       = "default"
	resourceName    = "example-foo"
	resourceGroup   = "samplecontroller.k8s.io"
	resourceVersion = "v1alpha1"
)

var (
	masterURL  string
	kubeconfig string
)

func main() {
	err := start()
	if err != nil {
		glog.Fatalf("error: %v", err)
		os.Exit(1)
	}

}
func start() error {
	flag.Parse()

	config, err := clientcmd.BuildConfigFromFlags(masterURL, kubeconfig)
	if err != nil {
		glog.Fatalf("Error building kubeconfig: %s", err.Error())
	}

	exampleClient, err := clientset.NewForConfig(config)
	if err != nil {
		glog.Fatalf("error build example clientset: %s", err.Error())
		return err
	}
	_, err = exampleClient.SamplecontrollerV1alpha1().Foos(namespace).Get(resourceName, metav1.GetOptions{})
	if err != nil {
		glog.Fatalf("err get example-foo: %s", err.Error())
		return err
	}

	//create a scale client for foo resource
	fooScaleClient, err := createNewScaleClient(config)
	if err != nil {
		glog.Fatalf("error build scale client: %s", err.Error())
	}
	groupResource := schema.GroupResource{
		Group:    resourceGroup,
		Resource: "foos",
	}
	// get the current scale sub resource
	gottenScale, err := fooScaleClient.Scales(namespace).Get(groupResource, resourceName)
	if err != nil {
		glog.Fatalf("error get the scale")
		return nil
	}
	glog.Infof("Got the scale subresource: %v", gottenScale)

	// scale the current replica +1
	currentNo := gottenScale.Spec.Replicas
	gottenScale.Spec.Replicas = currentNo + 1
	fooScaleClient.Scales(namespace).Update(groupResource, gottenScale)
	glog.Infof("Scale the replica from %v to %v using scale subresource", currentNo, gottenScale.Spec.Replicas)
	return nil
}

// This constructs a scale client that can be used against any resource that
// implements the 'scale' sub resource.
// It is special because it accepts a groupResource as part of its REST functions
func createNewScaleClient(config *rest.Config) (scale.ScalesGetter, error) {
	discoveryClient, err := discovery.NewDiscoveryClientForConfig(config)
	if err != nil {
		return nil, err
	}
	groupResource, err := discoveryClient.ServerResourcesForGroupVersion(resourceGroup + "/" + resourceVersion)
	if err != nil {
		return nil, err
	}

	resources := []*discovery.APIGroupResources{
		{
			Group: metav1.APIGroup{
				Name: resourceGroup,
				Versions: []metav1.GroupVersionForDiscovery{
					{Version: resourceVersion},
				},
				PreferredVersion: metav1.GroupVersionForDiscovery{Version: resourceVersion},
			},
			VersionedResources: map[string][]metav1.APIResource{
				resourceVersion: groupResource.APIResources,
			},
		},
	}

	restMapper := discovery.NewRESTMapper(resources, nil)
	// the resolver knows how to convert a groupVersion to its API path.
	resolver := scale.NewDiscoveryScaleKindResolver(discoveryClient)

	return scale.NewForConfig(config, restMapper, dynamic.LegacyAPIPathResolverFunc, resolver)
}

func init() {
	flag.StringVar(&kubeconfig, "kubeconfig", "", "Path to a kubeconfig. Only required if out-of-cluster.")
	flag.StringVar(&masterURL, "master", "", "The address of the Kubernetes API server. Overrides any value in kubeconfig. Only required if out-of-cluster.")
}
