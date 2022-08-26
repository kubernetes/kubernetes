/*
Copyright 2022 The KCP Authors.

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

package resourcequota

import (
	"github.com/kcp-dev/logicalcluster/v2"
	coreinformers "k8s.io/client-go/informers/core/v1"
)

// SetResourceQuotaInformer sets the lister and indexer on the quotaAccessor. This is used by kcp to inject a lister and
// indexer that are scoped to a single logical cluster. This replaces the need to use a.SetExternalKubeInformerFactory().
func (a *QuotaAdmission) SetResourceQuotaInformer(informer coreinformers.ResourceQuotaInformer) {
	a.quotaAccessor.lister = informer.Lister()
	a.quotaAccessor.indexer = informer.Informer().GetIndexer()
}

// SetClusterName sets the logicalcluster.Name on the quotaAccessor.
func (a *QuotaAdmission) SetClusterName(name logicalcluster.Name) {
	a.quotaAccessor.clusterName = name
}
