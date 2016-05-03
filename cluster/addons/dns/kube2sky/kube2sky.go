/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

// kube2sky is a bridge between Kubernetes and SkyDNS.  It watches the
// Kubernetes master for changes in Services and manifests them into etcd for
// SkyDNS to serve as DNS records.
package main

import (
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/golang/glog"
	flag "github.com/spf13/pflag"
	"k8s.io/kubernetes/cluster/addons/dns/kube2sky/lib"
	utilflag "k8s.io/kubernetes/pkg/util/flag"
)

var (
	argDomain              = flag.String("domain", "cluster.local", "domain under which to create names")
	argEtcdMutationTimeout = flag.Duration("etcd-mutation-timeout", 10*time.Second, "crash after retrying etcd mutation for a specified duration")
	argEtcdServer          = flag.String("etcd-server", "http://127.0.0.1:4001", "URL to etcd server")
	argKubecfgFile         = flag.String("kubecfg-file", "", "Location of kubecfg file for access to kubernetes master service; --kube-master-url overrides the URL part of this; if neither this nor --kube-master-url are provided, defaults to service account tokens")
	argKubeMasterURL       = flag.String("kube-master-url", "", "URL to reach kubernetes master. Env variables in this flag will be expanded.")
	healthzPort            = flag.Int("healthz-port", 8081, "port on which to serve a Kube2sky HTTP readiness probe.")
)

// setupSignalHandlers runs a goroutine that waits on SIGINT or SIGTERM and logs it
// before exiting.
func setupSignalHandlers() {
	sigChan := make(chan os.Signal)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	// This program should always exit gracefully logging that it received
	// either a SIGINT or SIGTERM. Since kube2sky is run in a container
	// without a liveness probe as part of the kube-dns pod, it shouldn't
	// restart unless the pod is deleted. If it restarts without logging
	// anything it means something is seriously wrong.
	// TODO: Remove once #22290 is fixed.
	go func() {
		glog.Fatalf("Received signal %s", <-sigChan)
	}()
}

// setupHealthzHandlers sets up a readiness and liveness endpoint for kube2sky.
func setupHealthzHandlers(ks *lib.Kube2sky) {
	http.HandleFunc("/readiness", func(w http.ResponseWriter, req *http.Request) {
		fmt.Fprintf(w, "ok\n")
	})
}

func main() {
	flag.CommandLine.SetNormalizeFunc(utilflag.WarnWordSepNormalizeFunc)
	flag.Parse()
	var err error
	setupSignalHandlers()
	// TODO: Validate input flags.
	domain := *argDomain
	if !strings.HasSuffix(domain, ".") {
		domain = fmt.Sprintf("%s.", domain)
	}
	ks := lib.Kube2sky{
		Domain:              domain,
		EtcdMutationTimeout: *argEtcdMutationTimeout,
	}
	if ks.EtcdClient, err = lib.NewEtcdClient(*argEtcdServer); err != nil {
		glog.Fatalf("Failed to create etcd client - %v", err)
	}

	kubeClient, err := lib.NewKubeClient(*argKubeMasterURL, *argKubecfgFile)
	if err != nil {
		glog.Fatalf("Failed to create a kubernetes client: %v", err)
	}
	// Wait synchronously for the Kubernetes service and add a DNS record for it.
	ks.NewService(lib.WaitForKubernetesService(kubeClient))
	glog.Infof("Successfully added DNS record for Kubernetes service.")

	ks.EndpointsStore = lib.WatchEndpoints(kubeClient, &ks)
	ks.ServicesStore = lib.WatchForServices(kubeClient, &ks)
	ks.PodsStore = lib.WatchPods(kubeClient, &ks)

	// We declare kube2sky ready when:
	// 1. It has retrieved the Kubernetes master service from the apiserver. If this
	//    doesn't happen skydns will fail its liveness probe assuming that it can't
	//    perform any cluster local DNS lookups.
	// 2. It has setup the 3 watches above.
	// Once ready this container never flips to not-ready.
	setupHealthzHandlers(&ks)
	glog.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", *healthzPort), nil))
}
