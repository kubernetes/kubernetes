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

// Package options provides the flags used for the controller manager.
//
// CAUTION: If you update code in this file, you may need to also update code
//          in contrib/mesos/pkg/controllermanager/controllermanager.go
package options

import (
	"time"

	"github.com/spf13/pflag"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/client/leaderelection"
)

type ControllerManagerConfiguration struct {
	unversioned.TypeMeta

	// port is the port that the controller-manager's http service runs on.
	Port int `json:"port"`
	// address is the IP address to serve on (set to 0.0.0.0 for all interfaces).
	Address string `json:"address"`
	// concurrentSubRCSyncs is the number of sub replication controllers that are
	// allowed to sync concurrently. Larger number = more responsive replica
	// management, but more CPU (and network) load.
	ConcurrentSubRCSyncs int `json:"concurrentSubRCSyncs"`
	// clusterMonitorPeriod is the period for syncing ClusterStatus in cluster controller.
	ClusterMonitorPeriod unversioned.Duration `json:"clusterMonitorPeriod"`
	// uberAPIQPS is the QPS to use while talking with ubernetes apiserver.
	UberAPIQPS float32 `json:"uberAPIQPS"`
	// uberAPIBurst is the burst to use while talking with ubernetes apiserver.
	UberAPIBurst int `json:"uberAPIBurst"`
	// enableProfiling enables profiling via web interface host:port/debug/pprof/
	EnableProfiling bool `json:"enableProfiling"`
	// leaderElection defines the configuration of leader election client.
	LeaderElection componentconfig.LeaderElectionConfiguration `json:"leaderElection"`
}

// CMServer is the main context object for the controller manager.
type CMServer struct {
	ControllerManagerConfiguration

	Master          string
	ApiServerconfig string
}

const (
	// UberControllerManagerPort is the default port for the ubernetes controller manager status server.
	// May be overridden by a flag at startup.
	UberControllerManagerPort = 10252
)

// NewCMServer creates a new CMServer with a default config.
func NewCMServer() *CMServer {
	s := CMServer{
		ControllerManagerConfiguration: ControllerManagerConfiguration{
			Port:                 UberControllerManagerPort,
			Address:              "0.0.0.0",
			ConcurrentSubRCSyncs: 5,
			ClusterMonitorPeriod: unversioned.Duration{40 * time.Second},
			UberAPIQPS:           20.0,
			UberAPIBurst:         30,
			LeaderElection:       leaderelection.DefaultLeaderElectionConfiguration(),
		},
	}
	return &s
}

// AddFlags adds flags for a specific CMServer to the specified FlagSet
func (s *CMServer) AddFlags(fs *pflag.FlagSet) {
	fs.IntVar(&s.Port, "port", s.Port, "The port that the controller-manager's http service runs on")
	fs.Var(componentconfig.IPVar{&s.Address}, "address", "The IP address to serve on (set to 0.0.0.0 for all interfaces)")
	fs.IntVar(&s.ConcurrentSubRCSyncs, "concurrent-subRc-syncs", s.ConcurrentSubRCSyncs, "The number of subRC syncing operations that will be done concurrently. Larger number = faster endpoint updating, but more CPU (and network) load")

	fs.DurationVar(&s.ClusterMonitorPeriod.Duration, "cluster-monitor-period", s.ClusterMonitorPeriod.Duration, "The period for syncing ClusterStatus in ClusterController.")
	fs.BoolVar(&s.EnableProfiling, "profiling", true, "Enable profiling via web interface host:port/debug/pprof/")
	fs.StringVar(&s.Master, "master", s.Master, "The address of the Kubernetes API server (overrides any value in kubeconfig)")
	fs.StringVar(&s.ApiServerconfig, "uberconfig", s.ApiServerconfig, "Path to ApiServerconfig file with authorization and master location information.")
	fs.Float32Var(&s.UberAPIQPS, "uber-api-qps", s.UberAPIQPS, "QPS to use while talking with ubernetes apiserver")
	fs.IntVar(&s.UberAPIBurst, "uber-api-burst", s.UberAPIBurst, "Burst to use while talking with ubernetes apiserver")
	leaderelection.BindFlags(&s.LeaderElection, fs)
}
