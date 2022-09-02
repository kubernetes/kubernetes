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

package phases

import (
	"io"

	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

// a package local type for testing purposes.
type testData struct{}

// testData must satisfy resetData.
var _ resetData = &testData{}

func (t *testData) ForceReset() bool                   { return false }
func (t *testData) InputReader() io.Reader             { return nil }
func (t *testData) IgnorePreflightErrors() sets.String { return nil }
func (t *testData) Cfg() *kubeadmapi.InitConfiguration { return nil }
func (t *testData) DryRun() bool                       { return false }
func (t *testData) Client() clientset.Interface        { return nil }
func (t *testData) CertificatesDir() string            { return "" }
func (t *testData) CRISocketPath() string              { return "" }
func (t *testData) CleanupTmpDir() bool                { return false }
