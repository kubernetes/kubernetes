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
	"github.com/spf13/pflag"

	"k8s.io/kubernetes/cmd/kube-dns/app/options"
	"k8s.io/kubernetes/pkg/dns"
	dnsconfig "k8s.io/kubernetes/pkg/dns/config"

	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
)

type KubeDNSServer struct {
	// DNS domain name.
	domain         string
	healthzPort    int
	dnsBindAddress string
	dnsPort        int
	kd             *dns.KubeDNS
}

func NewKubeDNSServerDefault(config *options.KubeDNSConfig) *KubeDNSServer {
	kubeClient, err := newKubeClient(config)
	if err != nil {
		glog.Fatalf("Failed to create a kubernetes client: %v", err)
	}

	var configSync dnsconfig.Sync
	if config.ConfigMap == "" {
		glog.V(0).Infof("ConfigMap not configured, using values from command line flags")
		configSync = dnsconfig.NewNopSync(
			&dnsconfig.Config{Federations: config.Federations})
	} else {
		glog.V(0).Infof("Using configuration read from ConfigMap: %v:%v",
			config.ConfigMapNs, config.ConfigMap)
		configSync = dnsconfig.NewSync(
			kubeClient, config.ConfigMapNs, config.ConfigMap)
	}

	return &KubeDNSServer{
		domain:         config.ClusterDomain,
		healthzPort:    config.HealthzPort,
		dnsBindAddress: config.DNSBindAddress,
		dnsPort:        config.DNSPort,
		kd:             dns.NewKubeDNS(kubeClient, config.ClusterDomain, config.InitialSyncTimeout, configSync),
	}
}

func newKubeClient(dnsConfig *options.KubeDNSConfig) (kubernetes.Interface, error) {
	var config *rest.Config
	var err error

	if dnsConfig.KubeConfigFile == "" {
		config, err = rest.InClusterConfig()
		if err != nil {
			return nil, err
		}
	} else {
		config, err = clientcmd.BuildConfigFromFlags(
			dnsConfig.KubeMasterURL, dnsConfig.KubeConfigFile)
		if err != nil {
			return nil, err
		}
	}

	return kubernetes.NewForConfig(config)
}

func (server *KubeDNSServer) Run() {
	pflag.VisitAll(func(flag *pflag.Flag) {
		glog.V(0).Infof("FLAG: --%s=%q", flag.Name, flag.Value)
	})
	setupSignalHandlers()
	server.startSkyDNSServer()
	server.kd.Start()
	server.setupHandlers()

	glog.V(0).Infof("Status HTTP port %v", server.healthzPort)
	glog.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", server.healthzPort), nil))
}

// setupHealthzHandlers sets up a readiness and liveness endpoint for kube2sky.
func (server *KubeDNSServer) setupHandlers() {
	glog.V(0).Infof("Setting up Healthz Handler (/readiness)")
	http.HandleFunc("/readiness", func(w http.ResponseWriter, req *http.Request) {
		fmt.Fprintf(w, "ok\n")
	})

	glog.V(0).Infof("Setting up cache handler (/cache)")
	http.HandleFunc("/cache", func(w http.ResponseWriter, req *http.Request) {
		serializedJSON, err := server.kd.GetCacheAsJSON()
		if err == nil {
			fmt.Fprint(w, serializedJSON)
		} else {
			w.WriteHeader(http.StatusInternalServerError)
			fmt.Fprint(w, err)
		}
	})
}

// setupSignalHandlers installs signal handler to ignore SIGINT and
// SIGTERM. This daemon will be killed by SIGKILL after the grace
// period to allow for some manner of graceful shutdown.
func setupSignalHandlers() {
	sigChan := make(chan os.Signal)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		glog.V(0).Infof("Ignoring signal %v (can only be terminated by SIGKILL)", <-sigChan)
	}()
}

func (d *KubeDNSServer) startSkyDNSServer() {
	glog.V(0).Infof("Starting SkyDNS server (%v:%v)", d.dnsBindAddress, d.dnsPort)
	skydnsConfig := &server.Config{
		Domain:  d.domain,
		DnsAddr: fmt.Sprintf("%s:%d", d.dnsBindAddress, d.dnsPort),
	}
	server.SetDefaults(skydnsConfig)
	s := server.New(d.kd, skydnsConfig)
	if err := metrics.Metrics(); err != nil {
		glog.Fatalf("Skydns metrics error: %s", err)
	} else if metrics.Port != "" {
		glog.V(0).Infof("Skydns metrics enabled (%v:%v)", metrics.Path, metrics.Port)
	} else {
		glog.V(0).Infof("Skydns metrics not enabled")
	}

	go s.Run()
}
