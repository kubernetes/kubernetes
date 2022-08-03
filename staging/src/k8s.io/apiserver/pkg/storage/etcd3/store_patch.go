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
	"regexp"
	"strings"

	"github.com/kcp-dev/logicalcluster/v2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/klog/v2"
)

var (
	// matches cluster/remainder, capturing cluster.
	//
	// Example: root:org:ws/some-namespace/some-name
	// Example: root:org:ws/some-name
	wildcardClusterNameRegex = regexp.MustCompile(`^([^/]+)\/.+`)

	// matches identity-or-customresources/cluster/remainder, capturing cluster
	//
	// Example: customresources/root:org:ws/some-namespace/some-name
	// Example: customresources/root:org:ws/some-name
	// Example: 2699f4d273d342adccdc8a32663408226ecf66de7d191113ed3d4dc9bccec2f2/root:org:ws/some-namespace/some-name
	// Example: 2699f4d273d342adccdc8a32663408226ecf66de7d191113ed3d4dc9bccec2f2/root:org:ws/some-name
	crdWildcardPartialMetadataClusterNameRegex = regexp.MustCompile(`^[^/]+\/([^/]+)\/.+`)
)

// adjustClusterNameIfWildcard determines the logical cluster name. If this is not a wildcard list/watch request,
// the cluster name is returned unmodified. Otherwise, the cluster name is extracted from the key based on whether
// it is a CR partial metadata request (prefix/identity/clusterName/remainder) or not (prefix/clusterName/remainder).
func adjustClusterNameIfWildcard(cluster *genericapirequest.Cluster, crdRequest bool, keyPrefix, key string) logicalcluster.Name {
	if cluster.Name != logicalcluster.Wildcard {
		return cluster.Name
	}

	keyWithoutPrefix := strings.TrimPrefix(key, keyPrefix)

	var regex *regexp.Regexp

	if cluster.PartialMetadataRequest && crdRequest {
		regex = crdWildcardPartialMetadataClusterNameRegex
	} else {
		regex = wildcardClusterNameRegex
	}

	matches := regex.FindStringSubmatch(keyWithoutPrefix)
	if len(matches) >= 2 {
		return logicalcluster.New(matches[1])
	}

	return logicalcluster.Name{}
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
