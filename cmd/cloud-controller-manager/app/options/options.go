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

package options

import (
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	cmoptions "k8s.io/kubernetes/cmd/controller-manager/app/options"
	"k8s.io/kubernetes/pkg/client/leaderelectionconfig"
	"k8s.io/kubernetes/pkg/master/ports"

	// add the kubernetes feature gates
	_ "k8s.io/kubernetes/pkg/features"

	"github.com/spf13/pflag"
)

// CloudControllerManagerServer is the main context object for the controller manager.
type CloudControllerManagerServer struct {
	cmoptions.ControllerManagerServer

	// NodeStatusUpdateFrequency is the frequency at which the controller updates nodes' status
	NodeStatusUpdateFrequency metav1.Duration
}

// NewCloudControllerManagerServer creates a new ExternalCMServer with a default config.
func NewCloudControllerManagerServer() *CloudControllerManagerServer {
	s := CloudControllerManagerServer{
		// The common/default are kept in 'cmd/kube-controller-manager/app/options/util.go'.
		// Please make common changes there and put anything cloud specific here.
		ControllerManagerServer: cmoptions.ControllerManagerServer{
			KubeControllerManagerConfiguration: cmoptions.GetDefaultControllerOptions(ports.CloudControllerManagerPort),
		},
		NodeStatusUpdateFrequency: metav1.Duration{Duration: 5 * time.Minute},
	}
	s.LeaderElection.LeaderElect = true
	return &s
}

// AddFlags adds flags for a specific ExternalCMServer to the specified FlagSet
func (s *CloudControllerManagerServer) AddFlags(fs *pflag.FlagSet) {
	cmoptions.AddDefaultControllerFlags(&s.ControllerManagerServer, fs)
	fs.StringVar(&s.CloudProvider, "cloud-provider", s.CloudProvider, "The provider of cloud services. Cannot be empty.")
	fs.DurationVar(&s.NodeStatusUpdateFrequency.Duration, "node-status-update-frequency", s.NodeStatusUpdateFrequency.Duration, "Specifies how often the controller updates nodes' status.")
	// TODO: remove --service-account-private-key-file 6 months after 1.8 is released (~1.10)
	fs.StringVar(&s.ServiceAccountKeyFile, "service-account-private-key-file", s.ServiceAccountKeyFile, "Filename containing a PEM-encoded private RSA or ECDSA key used to sign service account tokens.")
	fs.MarkDeprecated("service-account-private-key-file", "This flag is currently no-op and will be deleted.")
	fs.Int32Var(&s.ConcurrentServiceSyncs, "concurrent-service-syncs", s.ConcurrentServiceSyncs, "The number of services that are allowed to sync concurrently. Larger number = more responsive service management, but more CPU (and network) load")

	leaderelectionconfig.BindFlags(&s.LeaderElection, fs)

	utilfeature.DefaultFeatureGate.AddFlag(fs)
}
