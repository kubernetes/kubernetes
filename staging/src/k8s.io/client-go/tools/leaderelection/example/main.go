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
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/tools/leaderelection"
	"k8s.io/client-go/tools/leaderelection/resourcelock"
	"k8s.io/client-go/transport"
	"k8s.io/klog"
)

// main demonstrates a leader elected process that will step down if interrupted.
func main() {
	klog.InitFlags(nil)
	flag.Parse()
	args := flag.Args()
	if len(args) != 3 {
		log.Fatalf("requires three arguments: ID NAMESPACE CONFIG_MAP_NAME (%d)", len(args))
	}

	// leader election uses the Kubernetes API by writing to a ConfigMap or Endpoints
	// object. Conflicting writes are detected and each client handles those actions
	// independently.
	var config *rest.Config
	var err error
	if kubeconfig := os.Getenv("KUBECONFIG"); len(kubeconfig) > 0 {
		config, err = clientcmd.BuildConfigFromFlags("", kubeconfig)
	} else {
		config, err = rest.InClusterConfig()
	}
	if err != nil {
		log.Fatalf("failed to create client: %v", err)
	}

	// we use the ConfigMap lock type since edits to ConfigMaps are less common
	// and fewer objects in the cluster watch "all ConfigMaps" (unlike the older
	// Endpoints lock type, where quite a few system agents like the kube-proxy
	// and ingress controllers must watch endpoints).
	id := args[0]
	lock := &resourcelock.ConfigMapLock{
		ConfigMapMeta: metav1.ObjectMeta{
			Namespace: args[1],
			Name:      args[2],
		},
		Client: kubernetes.NewForConfigOrDie(config).CoreV1(),
		LockConfig: resourcelock.ResourceLockConfig{
			Identity: id,
		},
	}

	// use a Go context so we can tell the leaderelection code when we
	// want to step down
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// use a client that will stop allowing new requests once the context ends
	config.Wrap(transport.ContextCanceller(ctx, fmt.Errorf("the leader is shutting down")))
	exampleClient := kubernetes.NewForConfigOrDie(config).CoreV1()

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

	// start the leader election code loop
	leaderelection.RunOrDie(ctx, leaderelection.LeaderElectionConfig{
		Lock: lock,
		// IMPORTANT: you MUST ensure that any code you have that
		// is protected by the lease must terminate **before**
		// you call cancel. Otherwise, you could have a background
		// loop still running and another process could
		// get elected before your background loop finished, violating
		// the stated goal of the lease.
		ReleaseOnCancel: true,
		LeaseDuration:   60 * time.Second,
		RenewDeadline:   15 * time.Second,
		RetryPeriod:     5 * time.Second,
		Callbacks: leaderelection.LeaderCallbacks{
			OnStartedLeading: func(ctx context.Context) {
				// we're notified when we start - this is where you would
				// usually put your code
				log.Printf("%s: leading", id)
			},
			OnStoppedLeading: func() {
				// we can do cleanup here, or after the RunOrDie method
				// returns
				log.Printf("%s: lost", id)
			},
		},
	})

	// because the context is closed, the client should report errors
	_, err = exampleClient.ConfigMaps(args[1]).Get(args[2], metav1.GetOptions{})
	if err == nil || !strings.Contains(err.Error(), "the leader is shutting down") {
		log.Fatalf("%s: expected to get an error when trying to make a client call: %v", id, err)
	}

	// we no longer hold the lease, so perform any cleanup and then
	// exit
	log.Printf("%s: done", id)
}
