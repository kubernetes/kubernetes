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
	"bufio"
	"context"
	"flag"
	"fmt"
	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
	"k8s.io/client-go/util/retry"
	"os"
	"path/filepath"
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
	config, err := clientcmd.BuildConfigFromFlags("", *kubeconfig)
	if err != nil {
		panic(err)
	}
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		panic(err)
	}

	servicesClient := clientset.CoreV1().Services(apiv1.NamespaceDefault)

	svc := &apiv1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: "demo-service",
		},
		Spec: apiv1.ServiceSpec{
			Ports: []apiv1.ServicePort{
				{
					Name: "demo-port",
					Port: 80,
					TargetPort: intstr.IntOrString{
						Type:   0,
						IntVal: 80,
						StrVal: "",
					},
				},
			},
			Selector: map[string]string{
				"app": "demo",
			},
			Type: "LoadBalancer",
		},
	}

	// Create service
	fmt.Println("Creating service...")
	result, err := servicesClient.Create(context.TODO(), svc, metav1.CreateOptions{})
	if err != nil {
		panic(err)
	}
	fmt.Printf("Created service %q.\n", result.GetObjectMeta().GetName())

	// Update service
	prompt()
	fmt.Println("Updating service...")

	retryErr := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		result, getErr := servicesClient.Get(context.TODO(), "demo-service", metav1.GetOptions{})
		if getErr != nil {
			panic(fmt.Errorf("Failed to get latest version of Service: %v", getErr))
		}

		result.Spec.Type = apiv1.ServiceTypeNodePort
		_, updateErr := servicesClient.Update(context.TODO(), result, metav1.UpdateOptions{})
		return updateErr
	})
	if retryErr != nil {
		panic(fmt.Errorf("Update failed: %v", retryErr))
	}
	fmt.Println("Updated service...")

	// List services
	prompt()
	fmt.Printf("Listing services in namespace %q:\n", apiv1.NamespaceDefault)
	list, err := servicesClient.List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		panic(err)
	}
	for _, s := range list.Items {
		fmt.Printf(" * %s (%v replicas)\n", s.Name, s.Spec.Type)
	}

	// Delete service
	prompt()
	fmt.Println("Deleting service...")
	deletePolicy := metav1.DeletePropagationForeground
	if err := servicesClient.Delete(context.TODO(), "demo-service", metav1.DeleteOptions{
		PropagationPolicy: &deletePolicy,
	}); err != nil {
		panic(err)
	}
	fmt.Println("Deleted service.")
}

func prompt() {
	fmt.Printf("-> Press Return key to continue.")
	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		break
	}
	if err := scanner.Err(); err != nil {
		panic(err)
	}
	fmt.Println()
}
