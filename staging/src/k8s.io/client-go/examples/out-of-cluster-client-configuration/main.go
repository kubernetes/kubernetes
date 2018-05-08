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
	"os"
	"path/filepath"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	// Uncomment the following line to load the gcp plugin (only required to authenticate against GKE clusters).
	// _ "k8s.io/client-go/plugin/pkg/client/auth/gcp"
	"encoding/base64"
	"k8s.io/client-go/rest"
)

func main() {
	var config *rest.Config
	var err error

	if os.Getenv("KUBECONFIG_CONTENTS") == "" {
		config, err = loadConfigFromFile()
	} else {
		config, err = loadConfigFromEnvironment()
	}
	if err != nil {
		panic(err.Error())
	}

	// create the clientset
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		panic(err.Error())
	}
	for {
		pods, err := clientset.CoreV1().Pods("").List(metav1.ListOptions{})
		if err != nil {
			panic(err.Error())
		}
		fmt.Printf("There are %d pods in the cluster\n", len(pods.Items))

		// Examples for error handling:
		// - Use helper functions like e.g. errors.IsNotFound()
		// - And/or cast to StatusError and use its properties like e.g. ErrStatus.Message
		namespace := "default"
		pod := "example-xxxxx"
		_, err = clientset.CoreV1().Pods(namespace).Get(pod, metav1.GetOptions{})
		if errors.IsNotFound(err) {
			fmt.Printf("Pod %s in namespace %s not found\n", pod, namespace)
		} else if statusError, isStatus := err.(*errors.StatusError); isStatus {
			fmt.Printf("Error getting pod %s in namespace %s: %v\n",
				pod, namespace, statusError.ErrStatus.Message)
		} else if err != nil {
			panic(err.Error())
		} else {
			fmt.Printf("Found pod %s in namespace %s\n", pod, namespace)
		}

		time.Sleep(10 * time.Second)
	}
}

// This function will try to create a rest configuration by reading in a kubeconfig file stored on a filesystem
func loadConfigFromFile() (*rest.Config, error) {
	var kubeconfig *string
	if home := homeDir(); home != "" {
		// Offer to try to grab <home>/.kube/config by default since there is a known home directory
		kubeconfig = flag.String("kubeconfig", filepath.Join(home, ".kube", "config"), "(optional) absolute path to the kubeconfig file")
	} else {
		// No known home directory, so we will state that this is a required parameter
		kubeconfig = flag.String("kubeconfig", "", "absolute path to the kubeconfig file")
	}
	flag.Parse()

	fmt.Printf("Used filesystem-hosted kubeconfig for configuration")
	// load kubeconfig from this path and use the current context in kubeconfig
	return clientcmd.BuildConfigFromFlags("", *kubeconfig)
}

// This function will try to create a rest configuration by reading in an environmental variable
// Shows off how clientcmd.RESTConfigFromKubeConfig(...) can be used to connect to external kubernetes clusters
// through kubeconfigs delivered from an external service / not a file located on a filesystem
func loadConfigFromEnvironment() (*rest.Config, error) {
	// KUBECONFIG_CONTENTS is an environmental variable that is the contents of a kubeconfig file, base64 encoded
	// Setting data to the unencoded contents (should be a byte array that contains yaml)
	data, err := base64.StdEncoding.DecodeString(os.Getenv("KUBECONFIG_CONTENTS"))
	if err != nil {
		return nil, err
	}
	fmt.Printf("Used environmental variable for rest configuration\n")

	// Using the contents of the kubeconfig file, generate a rest config
	return clientcmd.RESTConfigFromKubeConfig(data)
}

func homeDir() string {
	if h := os.Getenv("HOME"); h != "" {
		return h
	}
	return os.Getenv("USERPROFILE") // windows
}
