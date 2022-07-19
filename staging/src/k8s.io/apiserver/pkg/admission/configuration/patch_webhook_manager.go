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

package configuration

import (
	"github.com/kcp-dev/logicalcluster/v2"

	"k8s.io/apiserver/pkg/admission/plugin/webhook"
)

type WebhookClusterAccessor interface {
	// GetLogicalCluster returns the logical cluster that provides this webhook.
	GetLogicalCluster() logicalcluster.Name
}

var _ WebhookClusterAccessor = &webhookClusterAccessorWrapper{}

// WithCluster attaches the logical cluster to the webhook.
func WithCluster(cluster logicalcluster.Name, webhook webhook.WebhookAccessor) webhook.WebhookAccessor {
	return webhookClusterAccessorWrapper{WebhookAccessor: webhook, cluster: cluster}
}

type webhookClusterAccessorWrapper struct {
	webhook.WebhookAccessor
	cluster logicalcluster.Name
}

func (w webhookClusterAccessorWrapper) GetLogicalCluster() logicalcluster.Name {
	return w.cluster
}
