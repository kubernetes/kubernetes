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

package etcd3

import (
	"strings"

	"github.com/kcp-dev/logicalcluster/v2"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"

	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
)

// adjustClusterNameIfWildcard determines the logical cluster name. If this is not a cluster-wildcard list/watch request,
// the cluster name is returned unmodified. Otherwise, the cluster name is extracted from the key based on whether it is
// - a shard-wildcard request: <prefix>/shardName/clusterName/<remainder>
// - CR partial metadata request: <prefix>/identity/clusterName/<remainder>
// - any other request: <prefix>/clusterName/<remainder>.
func adjustClusterNameIfWildcard(shard genericapirequest.Shard, cluster *genericapirequest.Cluster, crdRequest bool, keyPrefix, key string) logicalcluster.Name {
	if cluster.Name != logicalcluster.Wildcard && !cluster.Wildcard { // TODO: fix this duplicity, as well
		return cluster.Name
	}

	keyWithoutPrefix := strings.TrimPrefix(key, keyPrefix)
	parts := strings.SplitN(keyWithoutPrefix, "/", 3)

	extract := func(minLen, i int) logicalcluster.Name {
		if len(parts) < minLen {
			klog.Warningf("shard=%s cluster=%s invalid key=%s had %d parts, not %d", shard, cluster, keyWithoutPrefix, len(parts), minLen)
			return logicalcluster.Name{}
		}
		return logicalcluster.New(parts[i])
	}

	switch {
	case shard.Wildcard():
		// expecting shardName/clusterName/<remainder>
		return extract(3, 1)
	case cluster.PartialMetadataRequest && crdRequest:
		// expecting 2699f4d273d342adccdc8a32663408226ecf66de7d191113ed3d4dc9bccec2f2/root:org:ws/<remainder>
		// OR customresources/root:org:ws/<remainder>
		return extract(3, 1)
	default:
		// expecting root:org:ws/<remainder>
		return extract(2, 0)
	}
}

// setClusterNameOnDecodedObject applies clusterName to obj. This is necessary because we don't store the cluster
// name in the objects in storage. Instead, it is derived from the storage key, and then applied after retrieving
// the object from storage.
func setClusterNameOnDecodedObject(obj interface{}, clusterName logicalcluster.Name) {
	var s clusterNameSetter

	switch t := obj.(type) {
	case metav1.ObjectMetaAccessor:
		s = t.GetObjectMeta()
	case clusterNameSetter:
		s = t
	default:
		klog.Warningf("Could not set ClusterName %s on object: %T", clusterName, obj)
		return
	}

	annotations := s.GetAnnotations()
	if annotations == nil {
		annotations = make(map[string]string)
	}
	annotations[logicalcluster.AnnotationKey] = clusterName.String()
	s.SetAnnotations(annotations)
}

type clusterNameSetter interface {
	GetAnnotations() map[string]string
	SetAnnotations(a map[string]string)
}
