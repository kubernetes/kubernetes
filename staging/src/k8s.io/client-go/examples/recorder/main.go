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

package main

import (
	"flag"
	"fmt"
	"time"

	"github.com/golang/glog"

	meta_v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	core_v1 "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/pkg/api"
	"k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/tools/record"
)

func main() {
	var kubeconfig string
	var podName string
	var namespace string
	var server string
	flag.StringVar(&kubeconfig, "kubeconfig", "", "absolute path to the kubeconfig file")
	flag.StringVar(&server, "server", "", "API server url")
	flag.StringVar(&podName, "pod", "kube-apiserver-master", "Name of the pod")
	flag.StringVar(&namespace, "namespace", api.NamespaceSystem, "Namespace of the pod")
	flag.Parse()
	// uses the current context in kubeconfig
	config, err := clientcmd.BuildConfigFromFlags(server, kubeconfig)
	if err != nil {
		glog.Fatal(err)
	}
	// creates the clientset
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		glog.Fatal(err)
	}

	// Create a new broadcaster which will send events we generate to the apiserver
	broadcaster := record.NewBroadcaster()
	broadcaster.StartRecordingToSink(&core_v1.EventSinkImpl{Interface: clientset.Events(namespace)})
	recorder := broadcaster.NewRecorder(api.Scheme, v1.EventSource{Component: "pod-watcher", Host: "localhost"})

	// Create a watcher which will give us events from an object with the name of the Pod
	watcher, err := clientset.CoreV1().Events(namespace).
		Watch(meta_v1.ListOptions{FieldSelector: "involvedObject.name=" + podName})
	if err != nil {
		glog.Fatal(err)
	}

	defer watcher.Stop()

	// Watch for events associated with the specified pod name and print the output
	go func() {
		for obj := range watcher.ResultChan() {
			event := obj.Object.(*v1.Event)
			fmt.Printf("Reason: %s; Message: %s \n", event.Reason, event.Message)
		}
	}()

	// Every 10 seconds fetch the specified pod and emit a PodDiscovered event of severity `Normal`
	for {
		glog.V(2).Infof("Fetching pod %s", podName)
		pod, err := clientset.CoreV1().Pods(namespace).Get(podName, meta_v1.GetOptions{})
		if err != nil {
			glog.Fatal(err)
		}
		glog.V(2).Infof("Emitting event for pod %s", podName)
		recorder.Eventf(pod, "Normal", "PodDiscovered", "Discovered pod %s in namespace %s", podName, namespace)
		time.Sleep(10 * time.Second)
	}
}
