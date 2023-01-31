/*
Copyright 2022 The Kubernetes Authors.

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

package node

import (
	"io"

	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

// a package local type for testing purposes.
type testData struct{}

// testData must satisfy Data.
var _ Data = &testData{}

func (t *testData) EtcdUpgrade() bool                  { return false }
func (t *testData) RenewCerts() bool                   { return false }
func (t *testData) DryRun() bool                       { return false }
func (t *testData) Cfg() *kubeadmapi.InitConfiguration { return nil }
func (t *testData) IsControlPlaneNode() bool           { return false }
func (t *testData) Client() clientset.Interface        { return nil }
func (t *testData) IgnorePreflightErrors() sets.String { return nil }
func (t *testData) PatchesDir() string                 { return "" }
func (t *testData) KubeConfigPath() string             { return "" }
func (t *testData) OutputWriter() io.Writer            { return nil }
