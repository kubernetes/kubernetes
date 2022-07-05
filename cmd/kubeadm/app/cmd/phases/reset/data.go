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

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

// resetData is the interface to use for reset phases.
// The "resetData" type from "cmd/reset.go" must satisfy this interface.
type resetData interface {
	ForceReset() bool
	InputReader() io.Reader
	IgnorePreflightErrors() sets.String
	Cfg() *kubeadmapi.InitConfiguration
	DryRun() bool
	Client() clientset.Interface
	CertificatesDir() string
	CRISocketPath() string
}
