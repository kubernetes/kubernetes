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

package phases

import (
	"io"

	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"

	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

// a package local type for testing purposes.
type testInitData struct{}

// testInitData must satisfy InitData.
var _ InitData = &testInitData{}

func (t *testInitData) UploadCerts() bool                                    { return false }
func (t *testInitData) CertificateKey() string                               { return "" }
func (t *testInitData) SetCertificateKey(key string)                         {}
func (t *testInitData) SkipCertificateKeyPrint() bool                        { return false }
func (t *testInitData) Cfg() *kubeadmapi.InitConfiguration                   { return nil }
func (t *testInitData) DryRun() bool                                         { return false }
func (t *testInitData) SkipTokenPrint() bool                                 { return false }
func (t *testInitData) IgnorePreflightErrors() sets.Set[string]              { return nil }
func (t *testInitData) CertificateWriteDir() string                          { return "" }
func (t *testInitData) CertificateDir() string                               { return "" }
func (t *testInitData) KubeConfig() (*clientcmdapi.Config, error)            { return nil, nil }
func (t *testInitData) KubeConfigDir() string                                { return "" }
func (t *testInitData) KubeConfigPath() string                               { return "" }
func (t *testInitData) ManifestDir() string                                  { return "" }
func (t *testInitData) KubeletDir() string                                   { return "" }
func (t *testInitData) ExternalCA() bool                                     { return false }
func (t *testInitData) OutputWriter() io.Writer                              { return nil }
func (t *testInitData) Client() (clientset.Interface, error)                 { return nil, nil }
func (t *testInitData) ClientWithoutBootstrap() (clientset.Interface, error) { return nil, nil }
func (t *testInitData) Tokens() []string                                     { return nil }
func (t *testInitData) PatchesDir() string                                   { return "" }
