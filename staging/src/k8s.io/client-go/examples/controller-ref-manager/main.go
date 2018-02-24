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

package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"

	apps "k8s.io/api/apps/v1beta2"
	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	controllertools "k8s.io/client-go/tools/controller"
	// Uncomment the following line to load the gcp plugin (only required to authenticate against GKE clusters).
	// _ "k8s.io/client-go/plugin/pkg/client/auth/gcp"

	"k8s.io/client-go/examples/controller-ref-manager/controller"
)

var (
	productionLabel         = map[string]string{"type": "production", "app": "demo"}
	productionLabelSelector = labels.Set{"type": "production"}.AsSelector()
)

// RefManager normally used in controllers sync loop to adopt/release resources.
// See https://github.com/kubernetes/community/blob/master/contributors/design-proposals/api-machinery/controller-ref.md
// for more details.
func main() {
	kubeconfig := flag.String("kubeconfig", "", "absolute path to the kubeconfig file")
	master := flag.String("master", "", "master server URL")

	flag.Parse()

	config, err := clientcmd.BuildConfigFromFlags(*master, *kubeconfig)
	if err != nil {
		panic(err)
	}
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		panic(err)
	}

	myController := controller.MyController{ObjectMeta: metav1.ObjectMeta{Name: "demo"}}
	myController.UID = types.UID("123")

	deployment := &apps.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "demo-deployment",
			Labels: productionLabel,
		},
		Spec: apps.DeploymentSpec{
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "demo"}},
			Replicas: int32Ptr(2),
			Template: apiv1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"app": "demo",
					},
				},
				Spec: apiv1.PodSpec{
					Containers: []apiv1.Container{
						{
							Name:  "web",
							Image: "nginx:1.13",
							Ports: []apiv1.ContainerPort{
								{
									Name:          "http",
									Protocol:      apiv1.ProtocolTCP,
									ContainerPort: 80,
								},
							},
						},
					},
				},
			},
		},
	}

	// Create Deployment without OwnerReferences
	fmt.Println("Creating deployment...")
	deploymentsClient := clientset.AppsV1beta2().Deployments(apiv1.NamespaceDefault)
	result, err := deploymentsClient.Create(deployment)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Created deployment %q.\n", result.Name)
	fmt.Printf("You can check OwnerReferences using kubectl: \"kubectl get deployment %s -o yaml\" \n", result.Name)
	prompt()

	// If any adoptions are attempted, we should first recheck for deletion with
	// an uncached quorum (see #42639).
	canAdoptFunc := controllertools.RecheckDeletionTimestamp(func() (metav1.Object, error) {
		fresh, err := clientset.AppsV1beta1().Deployments(result.Namespace).Get(result.Name, metav1.GetOptions{})
		if err != nil {
			return nil, err
		}
		if fresh.UID != result.UID {
			return nil, fmt.Errorf("original Deployment %v/%v is gone: got uid %v, wanted %v",
				result.Namespace, result.Name, fresh.UID, result.UID)
		}
		return fresh, nil
	})

	gvk := schema.GroupVersionKind{Group: "examples.client-go.k8s.io", Version: "v1alpha1", Kind: "Example"}
	refManager := controller.NewDeploymentControllerRefManager(clientset, &myController, productionLabelSelector, gvk, canAdoptFunc)

	// Adopt deployment
	claimed, err := refManager.ClaimDeployments([]*apps.Deployment{result})
	if err != nil {
		panic(err)
	}

	for _, d := range claimed {
		fmt.Printf("Claimed deployment: %s\n", d.GetObjectMeta().GetName())
	}

	fmt.Printf("You can check OwnerReferences using kubectl: \"kubectl get deployment %s -o yaml\" \n", result.GetObjectMeta().GetName())
	prompt()

	fmt.Println("Deleting deployment...")
	deletePolicy := metav1.DeletePropagationForeground
	err = deploymentsClient.Delete("demo-deployment", &metav1.DeleteOptions{PropagationPolicy: &deletePolicy})
	if err != nil {
		panic(err)
	}

	fmt.Println("Done")
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

func int32Ptr(i int32) *int32 { return &i }
