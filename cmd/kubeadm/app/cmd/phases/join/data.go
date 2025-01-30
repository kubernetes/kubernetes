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

// Package phases includes command line phases for kubeadm join
package phases

import (
	"io"

	"github.com/pkg/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
)

// JoinData is the interface to use for join phases.
// The "joinData" type from "cmd/join.go" must satisfy this interface.
type JoinData interface {
	CertificateKey() string
	Cfg() *kubeadmapi.JoinConfiguration
	TLSBootstrapCfg() (*clientcmdapi.Config, error)
	InitCfg() (*kubeadmapi.InitConfiguration, error)
	Client() (clientset.Interface, error)
	IgnorePreflightErrors() sets.Set[string]
	OutputWriter() io.Writer
	PatchesDir() string
	DryRun() bool
	KubeConfigDir() string
	KubeletDir() string
	ManifestDir() string
	CertificateWriteDir() string
}

func checkFeatureState(c workflow.RunData, featureGate string, state bool) (bool, error) {
	data, ok := c.(JoinData)
	if !ok {
		return false, errors.New("control-plane-join phase invoked with an invalid data struct")
	}

	cfg, err := data.InitCfg()
	if err != nil {
		return false, err
	}

	return state == features.Enabled(cfg.FeatureGates, featureGate), nil
}
