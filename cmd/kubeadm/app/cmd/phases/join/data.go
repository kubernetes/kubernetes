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

	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

// JoinData is the interface to use for join phases.
// The "joinData" type from "cmd/join.go" must satisfy this interface.
type JoinData interface {
	CertificateKey() string
	Cfg() *kubeadmapi.JoinConfiguration
	TLSBootstrapCfg() (*clientcmdapi.Config, error)
	InitCfg() (*kubeadmapi.InitConfiguration, error)
	Client() (clientset.Interface, error)
	WaitControlPlaneClient() (clientset.Interface, error)
	IgnorePreflightErrors() sets.Set[string]
	OutputWriter() io.Writer
	PatchesDir() string
	DryRun() bool
	KubeConfigDir() string
	KubeletDir() string
	ManifestDir() string
	CertificateWriteDir() string
}

// a package local type for testing purposes.
type testJoinData struct{}

// testJoinData must satisfy JoinData.
var _ JoinData = &testJoinData{}

func (j *testJoinData) CertificateKey() string                               { return "" }
func (j *testJoinData) Cfg() *kubeadmapi.JoinConfiguration                   { return nil }
func (j *testJoinData) TLSBootstrapCfg() (*clientcmdapi.Config, error)       { return nil, nil }
func (j *testJoinData) InitCfg() (*kubeadmapi.InitConfiguration, error)      { return nil, nil }
func (j *testJoinData) Client() (clientset.Interface, error)                 { return nil, nil }
func (j *testJoinData) WaitControlPlaneClient() (clientset.Interface, error) { return nil, nil }
func (j *testJoinData) IgnorePreflightErrors() sets.Set[string]              { return nil }
func (j *testJoinData) OutputWriter() io.Writer                              { return nil }
func (j *testJoinData) PatchesDir() string                                   { return "" }
func (j *testJoinData) DryRun() bool                                         { return false }
func (j *testJoinData) KubeConfigDir() string                                { return "" }
func (j *testJoinData) KubeletDir() string                                   { return "" }
func (j *testJoinData) ManifestDir() string                                  { return "" }
func (j *testJoinData) CertificateWriteDir() string                          { return "" }
