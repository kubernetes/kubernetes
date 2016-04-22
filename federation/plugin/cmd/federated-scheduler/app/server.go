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

// Package app implements a Server object for running the federated-scheduler.
package app

import (
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/pprof"
	"os"
	"strconv"

	"k8s.io/kubernetes/federation/client/clientset_generated/release_1_3"
	"k8s.io/kubernetes/pkg/api"
	kubeclient "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_3"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	"k8s.io/kubernetes/pkg/healthz"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/configz"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler"
	"k8s.io/kubernetes/federation/plugin/cmd/federated-scheduler/app/options"
	_ "k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/algorithmprovider"
	schedulerapi "k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/api"
	latestschedulerapi "k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/api/latest"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/factory"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
)

// NewSchedulerCommand creates a *cobra.Command object with default parameters
func NewSchedulerCommand() *cobra.Command {
	s := options.NewSchedulerServer()
	s.AddFlags(pflag.CommandLine)
	cmd := &cobra.Command{
		Use: "federated-scheduler",
		Long: `The Federation federated-scheduler is a policy-rich, topology-aware,
workload-specific function that significantly impacts availability, performance,
and capacity. The federated-scheduler needs to take into account individual and collective
resource requirements, quality of service requirements, hardware/software/policy
constraints, affinity and anti-affinity specifications, data locality, inter-workload
interference, deadlines, and so on. Workload-specific requirements will be exposed
through the API as necessary.`,
		Run: func(cmd *cobra.Command, args []string) {
		},
	}

	return cmd
}

// Run runs the specified SchedulerServer.  This should never exit.
func Run(s *options.SchedulerServer) error {
	if c, err := configz.New("componentconfig"); err == nil {
		c.Set(s.KubeSchedulerConfiguration)
	} else {
		glog.Errorf("unable to register configz: %s", err)
	}
	kubeconfig, err := clientcmd.BuildConfigFromFlags(s.Master, s.Kubeconfig)
	if err != nil {
		return err
	}

	// Override kubeconfig qps/burst settings from flags
	kubeconfig.QPS = s.KubeAPIQPS
	kubeconfig.Burst = s.KubeAPIBurst
	federatedClientSet := release_1_3.NewForConfigOrDie(kubeconfig)
	kubeClientSet := kubeclient.NewForConfigOrDie(kubeconfig)

	go func() {
		mux := http.NewServeMux()
		healthz.InstallHandler(mux)
		if s.EnableProfiling {
			mux.HandleFunc("/debug/pprof/", pprof.Index)
			mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
			mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
		}
		configz.InstallHandler(mux)
		mux.Handle("/metrics", prometheus.Handler())

		server := &http.Server{
			Addr:    net.JoinHostPort(s.Address, strconv.Itoa(s.Port)),
			Handler: mux,
		}
		glog.Fatal(server.ListenAndServe())
	}()

	configFactory := factory.NewConfigFactory(federatedClientSet, kubeClientSet, s.SchedulerName)
	config, err := createConfig(s, configFactory)

	if err != nil {
		glog.Fatalf("Failed to create federated-scheduler configuration: %v", err)
	}

	eventBroadcaster := record.NewBroadcaster()
	config.Recorder = eventBroadcaster.NewRecorder(api.EventSource{Component: s.SchedulerName})
	eventBroadcaster.StartLogging(glog.Infof)
	//TODO log event?
	//eventBroadcaster.StartRecordingToSink(federationClient.Events(""))

	sched := scheduler.New(config)

	run := func(_ <-chan struct{}) {
		sched.Run()
		select {}
	}
	run(nil)
	glog.Fatal("this statement is unreachable")
	panic("unreachable")
}

func createConfig(s *options.SchedulerServer, configFactory *factory.ConfigFactory) (*scheduler.Config, error) {
	//keep the branch but in Federation Phase I, CreateFromConfig is not needed as we do not have customized plugins.
	if _, err := os.Stat(s.PolicyConfigFile); err == nil {
		var (
			policy     schedulerapi.Policy
			configData []byte
		)
		configData, err := ioutil.ReadFile(s.PolicyConfigFile)
		if err != nil {
			return nil, fmt.Errorf("unable to read policy config: %v", err)
		}
		if err := runtime.DecodeInto(latestschedulerapi.Codec, configData, &policy); err != nil {
			return nil, fmt.Errorf("invalid configuration: %v", err)
		}
		return configFactory.CreateFromConfig(policy)
	}

	// if the config file isn't provided, use the specified (or default) provider
	// check of algorithm provider is registered and fail fast
	_, err := factory.GetAlgorithmProvider(s.AlgorithmProvider)
	if err != nil {
		return nil, err
	}

	return configFactory.CreateFromProvider(s.AlgorithmProvider)
}
