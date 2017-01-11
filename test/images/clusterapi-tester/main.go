/*
Copyright 2014 The Kubernetes Authors.

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

// A simple pod that first lists all nodes/services through the Kubernetes
// api, then serves a 200 on /healthz.
package main

import (
	"log"

	"fmt"
	"net/http"

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/kubernetes/pkg/api"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/fields"
)

func main() {
	cc, err := restclient.InClusterConfig()
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	kubeClient, err := clientset.NewForConfig(cc)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}
	listAll := api.ListOptions{LabelSelector: labels.Everything(), FieldSelector: fields.Everything()}
	nodes, err := kubeClient.Core().Nodes().List(listAll)
	if err != nil {
		log.Fatalf("Failed to list nodes: %v", err)
	}
	log.Printf("Nodes:")
	for _, node := range nodes.Items {
		log.Printf("\t%v", node.Name)
	}
	services, err := kubeClient.Core().Services(api.NamespaceDefault).List(listAll)
	if err != nil {
		log.Fatalf("Failed to list services: %v", err)
	}
	log.Printf("Services:")
	for _, svc := range services.Items {
		log.Printf("\t%v", svc.Name)
	}
	log.Printf("Success")
	http.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "Ok")
	})
	log.Fatal(http.ListenAndServe(":8080", nil))
}
