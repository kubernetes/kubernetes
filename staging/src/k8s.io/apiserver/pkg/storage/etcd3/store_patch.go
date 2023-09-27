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

	"github.com/kcp-dev/logicalcluster/v3"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/klog/v2"
)

// adjustClusterNameIfWildcard determines the logical cluster name from the key: <prefix>/clusterName/<remainder>.
func adjustClusterNameIfWildcard(cluster *genericapirequest.Cluster, keyPrefix, key string) logicalcluster.Name {
	if !cluster.Wildcard {
		return cluster.Name
	}

	keyWithoutPrefix := strings.TrimPrefix(key, keyPrefix)
	parts := strings.SplitN(keyWithoutPrefix, "/", 3)

	extract := func(minLen, i int) logicalcluster.Name {
		if len(parts) < minLen {
			klog.Warningf("shard=%s cluster=%s invalid key=%s had %d parts, not %d", shard, cluster, keyWithoutPrefix, len(parts), minLen)
			return ""
		}
		return logicalcluster.Name(parts[i])
	}

	// expecting root:org:ws/<remainder>
	return extract(2, 0)
}

// adjustShardNameIfWildcard determines a shard name. If this is not a shard-wildcard request,
// the shard name is returned unmodified. Otherwise, the shard name is extracted from the storage key.
func adjustShardNameIfWildcard(shard genericapirequest.Shard, keyPrefix, key string) genericapirequest.Shard {
	if !shard.Empty() && !shard.Wildcard() {
		return shard
	}

	if !shard.Wildcard() {
		// no-op: we can only assign shard names
		// to a request that explicitly asked for it
		return ""
	}

	keyWithoutPrefix := strings.TrimPrefix(key, keyPrefix)
	parts := strings.SplitN(keyWithoutPrefix, "/", 3)
	if len(parts) < 3 {
		klog.Warningf("unable to extract a shard name, invalid key=%s had %d parts, not %d", keyWithoutPrefix, len(parts), 3)
		return ""
	}
	return genericapirequest.Shard(parts[0])
}

// annotateDecodedObjectWith applies clusterName and shardName to an object.
// This is necessary because we don't store the cluster name and the shard name in the objects in storage.
// Instead, they are derived from the storage key, and then applied after retrieving the object from storage.
func annotateDecodedObjectWith(obj interface{}, clusterName logicalcluster.Name, shardName genericapirequest.Shard) {
	var s nameSetter

	switch t := obj.(type) {
	case metav1.ObjectMetaAccessor:
		s = t.GetObjectMeta()
	case nameSetter:
		s = t
	default:
		klog.Warningf("Could not set ClusterName %s, ShardName %s on object: %T", clusterName, shardName, obj)
		return
	}

	annotations := s.GetAnnotations()
	if annotations == nil {
		annotations = make(map[string]string)
	}
	annotations[logicalcluster.AnnotationKey] = clusterName.String()
	if !shardName.Empty() {
		annotations[genericapirequest.AnnotationKey] = shardName.String()
	}
	s.SetAnnotations(annotations)
}

type nameSetter interface {
	GetAnnotations() map[string]string
	SetAnnotations(a map[string]string)
}
