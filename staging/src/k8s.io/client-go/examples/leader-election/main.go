/*
Copyright 2019 The Kubernetes Authors.

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
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/tools/leaderelection"
	"k8s.io/client-go/tools/leaderelection/resourcelock"
	"k8s.io/client-go/transport"
	"k8s.io/klog"
)

func buildConfig(kubeconfig string) (*rest.Config, error) {
	if kubeconfig != "" {
		cfg, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
		if err != nil {
			return nil, err
		}
		return cfg, nil
	}

	cfg, err := rest.InClusterConfig()
	if err != nil {
		return nil, err
	}
	return cfg, nil
}

func main() {
	klog.InitFlags(nil)

	var kubeconfig string
	var resourceName string
	var ttlSeconds int

	flag.StringVar(&kubeconfig, "kubeconfig", "", "absolute path to the kubeconfig file")
	flag.StringVar(&resourceName, "resource-name", "example", "The leader election resource name (Configmap or Endpoint name)")
	flag.IntVar(&ttlSeconds, "ttl", 30, "TTL for leader election in seconds")
	flag.Parse()

	podName := os.Getenv("POD_NAME")
	podNamespace := os.Getenv("POD_NAMESPACE")

	if podName == "" || podNamespace == "" {
		klog.Fatal("unable to get POD information (missing POD_NAME or POD_NAMESPACE environment variable")
	}

	// leader election uses the Kubernetes API by writing to a ConfigMap or Endpoints
	// object. Conflicting writes are detected and each client handles those actions
	// independently.
	config, err := buildConfig(kubeconfig)
	if err != nil {
		klog.Fatal(err)
	}
	client := clientset.NewForConfigOrDie(config)

	// We use the ConfigMap lock type since edits to ConfigMaps are less common
	// and fewer objects in the cluster watch "all ConfigMaps" (unlike the older
	// Endpoints lock type, where quite a few system agents like the kube-proxy
	// and ingress controllers must watch endpoints).
	lock := &resourcelock.ConfigMapLock{
		ConfigMapMeta: metav1.ObjectMeta{
			Namespace: podNamespace,
			Name:      resourceName,
		},
		Client: client.CoreV1(),
		LockConfig: resourcelock.ResourceLockConfig{
			Identity: podName,
		},
	}

	// use a Go context so we can tell the leaderelection code when we
	// want to step down
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// use a client that will stop allowing new requests once the context ends
	config.Wrap(transport.ContextCanceller(ctx, fmt.Errorf("the leader is shutting down")))

	// listen for interrupts or the Linux SIGTERM signal and cancel
	// our context, which the leader election code will observe and
	// step down
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-ch
		log.Printf("Received termination, signaling shutdown")
		cancel()
	}()

	// events from Kubernetes
	callbacks := leaderelection.LeaderCallbacks{
		OnStartedLeading: func(ctx context.Context) {
			// we're notified when we start - this is where you would
			// usually put your code
			klog.Infof("%s: leading", podName)
		},
		OnStoppedLeading: func() {
			// we can do cleanup here, or after the RunOrDie method
			// returns
			klog.Infof("%s: lost", podName)
		},
		OnNewLeader: func(identity string) {
			// we're notified when new leader elected
			if identity == podName {
				// I just got the lock
				return
			}
			klog.Infof("new leader elected: %v", identity)
		},
	}

	// start the leader election code loop
	ttl := time.Duration(ttlSeconds) * time.Second
	leaderelection.RunOrDie(ctx, leaderelection.LeaderElectionConfig{
		Lock: lock,
		// IMPORTANT: you MUST ensure that any code you have that
		// is protected by the lease must terminate **before**
		// you call cancel. Otherwise, you could have a background
		// loop still running and another process could
		// get elected before your background loop finished, violating
		// the stated goal of the lease.
		ReleaseOnCancel: true,
		LeaseDuration:   ttl,
		RenewDeadline:   ttl / 2,
		RetryPeriod:     ttl / 4,
		Callbacks:       callbacks,
	})

	// because the context is closed, the client should report errors
	_, err = client.CoreV1().ConfigMaps(podNamespace).Get(resourceName, metav1.GetOptions{})
	if err == nil || !strings.Contains(err.Error(), "the leader is shutting down") {
		log.Fatalf("%s: expected to get an error when trying to make a client call: %v", podName, err)
	}

	// we no longer hold the lease, so perform any cleanup and then
	// exit
	log.Printf("%s: done", podName)
}
