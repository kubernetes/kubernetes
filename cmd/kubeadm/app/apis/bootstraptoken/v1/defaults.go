/*
Copyright 2023 The Kubernetes Authors.

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

package v1

import (
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	bootstrapapi "k8s.io/cluster-bootstrap/token/api"
)

const (
	// DefaultTokenDuration specifies the default amount of time that a bootstrap token will be valid
	// Default behaviour is 24 hours
	DefaultTokenDuration = 24 * time.Hour
)

var (
	// DefaultTokenUsages specifies the default functions a token will get
	DefaultTokenUsages = bootstrapapi.KnownTokenUsages

	// DefaultTokenGroups specifies the default groups that this token will authenticate as when used for authentication
	DefaultTokenGroups = []string{"system:bootstrappers:kubeadm:default-node-token"}
)

// SetDefaults_BootstrapToken sets the defaults for an individual Bootstrap Token
func SetDefaults_BootstrapToken(bt *BootstrapToken) {
	if bt.TTL == nil {
		bt.TTL = &metav1.Duration{
			Duration: DefaultTokenDuration,
		}
	}
	if len(bt.Usages) == 0 {
		bt.Usages = DefaultTokenUsages
	}

	if len(bt.Groups) == 0 {
		bt.Groups = DefaultTokenGroups
	}
}
