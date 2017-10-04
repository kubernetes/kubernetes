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
	"flag"
	"fmt"
	"os"
	"path/filepath"

	appsv1beta1 "k8s.io/api/apps/v1beta1"
	apiv1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	// Uncomment the following line to load the gcp plugin (only required to authenticate against GKE clusters).
	// _ "k8s.io/client-go/plugin/pkg/client/auth/gcp"
)

func main() {
	var kubeconfig *string
	if home := homeDir(); home != "" {
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

	deploymentsClient := clientset.AppsV1beta1().Deployments(apiv1.NamespaceDefault)

	deployment := &appsv1beta1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name: "demo-deployment",
		},
		Spec: appsv1beta1.DeploymentSpec{
			Replicas: int32Ptr(2),
			Template: apiv1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"app": "demo",
					},
					Annotations: map[string]string{
						"fizz": "buzz",
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

	// Create Deployment
	fmt.Println("Creating deployment...")
	result, err := deploymentsClient.Create(deployment)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Created deployment %q.\n", result.GetObjectMeta().GetName())

	// Update Deployment
	prompt()
	fmt.Println("Updating deployment...")
	//    You have two options to Update() this Deployment:
	//
	//    1. Modify the "deployment" variable and call: Update(deployment).
	//       This works like the "kubectl replace" command and it overwrites/loses changes
	//       made by other clients between you Create() and Update() the object.
	//    2. Modify the "result" returned by Create()/Get() and retry Update(result) until
	//       you no longer get a conflict error. This way, you can preserve changes made
	//       by other clients between Create() and Update(). This is implemented below:
	//
	// See the API Conventions:
	// https://github.com/kubernetes/community/blob/master/contributors/devel/api-conventions.md#concurrency-control-and-consistency

	for {
		// Retrieve latest version of Deployment before modifying and updating
		result, err = deploymentsClient.Get("demo-deployment", metav1.GetOptions{})
		if err != nil {
			panic(fmt.Errorf("Get failed: %+v", err))
		}
		result.Spec.Replicas = int32Ptr(1)                    // reduce replica count
		result.Spec.Template.Annotations = map[string]string{ // add annotations
			"foo": "bar",
		}

		if _, err = deploymentsClient.Update(result); err != nil {
			if errors.IsConflict(err) {
				// Deployment was modified since last retrieved, need to retry
				fmt.Println("Conflicting resource versions, retrying")
			} else {
				panic(err)
			}
		} else {
			fmt.Println("Updated deployment...")
			break
		}

		// TODO: You should sleep here with an exponential backoff to avoid
		// exhausting the apiserver, and add a limit/timeout on the retries to
		// avoid getting stuck in this loop indefintiely.
	}

	// Rollback Deployment
	prompt()
	fmt.Println("Rolling back deployment...")

	// Use same method as above to avoid version conflicts
	for {
		result, err = deploymentsClient.Get("demo-deployment", metav1.GetOptions{})
		if err != nil {
			panic(fmt.Errorf("Get failed: %+v", err))
		}
		result.Spec.RollbackTo = &appsv1beta1.RollbackConfig{
			Revision: 0, // Can be specific revision number or 0 for last revision
		}

		if _, err = deploymentsClient.Update(result); err != nil {
			if errors.IsConflict(err) {
				fmt.Println("Conflicting resource versions, retrying")
			} else {
				panic(err)
			}
		} else {
			fmt.Println("Rolled back deployment...")
			break
		}
	}

	// List Deployments
	prompt()
	fmt.Printf("Listing deployments in namespace %q:\n", apiv1.NamespaceDefault)
	list, err := deploymentsClient.List(metav1.ListOptions{})
	if err != nil {
		panic(err)
	}
	for _, d := range list.Items {
		fmt.Printf(" * %s (%d replicas)\n", d.Name, *d.Spec.Replicas)
	}

	// Delete Deployment
	prompt()
	fmt.Println("Deleting deployment...")
	deletePolicy := metav1.DeletePropagationForeground
	if err := deploymentsClient.Delete("demo-deployment", &metav1.DeleteOptions{
		PropagationPolicy: &deletePolicy,
	}); err != nil {
		panic(err)
	}
	fmt.Println("Deleted deployment.")
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

func homeDir() string {
	if h := os.Getenv("HOME"); h != "" {
		return h
	}
	return os.Getenv("USERPROFILE") // windows
}
