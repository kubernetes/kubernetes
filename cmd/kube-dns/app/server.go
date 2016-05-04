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

package app

import (
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"github.com/golang/glog"

	"github.com/skynetservices/skydns/metrics"
	"github.com/skynetservices/skydns/server"
	"k8s.io/kubernetes/cmd/kube-dns/app/options"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/restclient"
	kclient "k8s.io/kubernetes/pkg/client/unversioned"
	kclientcmd "k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	kdns "k8s.io/kubernetes/pkg/dns"
)

type KubeDNSServer struct {
	// DNS domain name.
	domain      string
	healthzPort int
	kd          *kdns.KubeDNS
}

func NewKubeDNSServerDefault(config *options.KubeDNSConfig) *KubeDNSServer {
	ks := KubeDNSServer{
		domain: config.ClusterDomain,
	}

	kubeClient, err := newKubeClient(config)
	if err != nil {
		glog.Fatalf("Failed to create a kubernetes client: %v", err)
	}
	ks.healthzPort = config.HealthzPort
	ks.kd = kdns.NewKubeDNS(kubeClient, config.ClusterDomain)
	return &ks
}

// TODO: evaluate using pkg/client/clientcmd
func newKubeClient(dnsConfig *options.KubeDNSConfig) (*kclient.Client, error) {
	var (
		config *restclient.Config
		err    error
	)

	if dnsConfig.KubeMasterURL != "" && dnsConfig.KubeConfigFile == "" {
		// Only --kube-master-url was provided.
		config = &restclient.Config{
			Host:          dnsConfig.KubeMasterURL,
			ContentConfig: restclient.ContentConfig{GroupVersion: &unversioned.GroupVersion{Version: "v1"}},
		}
	} else {
		// We either have:
		//  1) --kube-master-url and --kubecfg-file
		//  2) just --kubecfg-file
		//  3) neither flag
		// In any case, the logic is the same.  If (3), this will automatically
		// fall back on the service account token.
		overrides := &kclientcmd.ConfigOverrides{}
		overrides.ClusterInfo.Server = dnsConfig.KubeMasterURL                                // might be "", but that is OK
		rules := &kclientcmd.ClientConfigLoadingRules{ExplicitPath: dnsConfig.KubeConfigFile} // might be "", but that is OK
		if config, err = kclientcmd.NewNonInteractiveDeferredLoadingClientConfig(rules, overrides).ClientConfig(); err != nil {
			return nil, err
		}
	}

	glog.Infof("Using %s for kubernetes master", config.Host)
	glog.Infof("Using kubernetes API %v", config.GroupVersion)
	return kclient.New(config)
}

func (server *KubeDNSServer) Run() {
	setupSignalHandlers()
	server.startSkyDNSServer()
	server.kd.Start()
	server.setupHealthzHandlers()
	glog.Infof("Setting up Healthz Handler(/readiness, /cache) on port :%d", server.healthzPort)
	glog.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", server.healthzPort), nil))
}

// setupHealthzHandlers sets up a readiness and liveness endpoint for kube2sky.
func (server *KubeDNSServer) setupHealthzHandlers() {
	http.HandleFunc("/readiness", func(w http.ResponseWriter, req *http.Request) {
		fmt.Fprintf(w, "ok\n")
	})
	http.HandleFunc("/cache", func(w http.ResponseWriter, req *http.Request) {
		fmt.Fprint(w, server.kd.GetCacheAsJSON())
	})
}

// setupSignalHandlers runs a goroutine that waits on SIGINT or SIGTERM and logs it
// before exiting.
func setupSignalHandlers() {
	sigChan := make(chan os.Signal)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		glog.Fatalf("Received signal: %s", <-sigChan)
	}()
}

func (d *KubeDNSServer) startSkyDNSServer() {
	skydnsConfig := &server.Config{Domain: d.domain, DnsAddr: "0.0.0.0:53"}
	server.SetDefaults(skydnsConfig)
	s := server.New(d.kd, skydnsConfig)
	if err := metrics.Metrics(); err != nil {
		glog.Fatalf("skydns: %s", err)
	} else {
		glog.Infof("skydns: metrics enabled on :%s%s", metrics.Port, metrics.Path)
	}
	go s.Run()
}
