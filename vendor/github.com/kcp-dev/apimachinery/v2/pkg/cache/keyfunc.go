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

package cache

import (
	"fmt"
	"strings"

	"github.com/kcp-dev/logicalcluster/v3"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/client-go/tools/cache"
)

// DeletionHandlingMetaClusterNamespaceKeyFunc checks for
// DeletedFinalStateUnknown objects before calling
// MetaClusterNamespaceKeyFunc.
func DeletionHandlingMetaClusterNamespaceKeyFunc(obj interface{}) (string, error) {
	if d, ok := obj.(cache.DeletedFinalStateUnknown); ok {
		return d.Key, nil
	}
	return MetaClusterNamespaceKeyFunc(obj)
}

// MetaClusterNamespaceKeyFunc is a convenient default KeyFunc which knows how to make
// keys for API objects which implement meta.Interface.
// The key uses the format <clusterName>|<namespace>/<name> unless <namespace> is empty, then
// it's just <clusterName>|<name>, and if running in a single-cluster context where no explicit
// cluster name is given, it's just <name>.
func MetaClusterNamespaceKeyFunc(obj interface{}) (string, error) {
	if key, ok := obj.(cache.ExplicitKey); ok {
		return string(key), nil
	}
	meta, err := meta.Accessor(obj)
	if err != nil {
		return "", fmt.Errorf("object has no meta: %v", err)
	}
	return ToClusterAwareKey(logicalcluster.From(meta).String(), meta.GetNamespace(), meta.GetName()), nil
}

// ToClusterAwareKey formats a cluster, namespace, and name as a key.
func ToClusterAwareKey(cluster, namespace, name string) string {
	var key string
	if cluster != "" {
		key += cluster + "|"
	}
	if namespace != "" {
		key += namespace + "/"
	}
	key += name
	return key
}

// SplitMetaClusterNamespaceKey returns the namespace and name that
// MetaClusterNamespaceKeyFunc encoded into key.
func SplitMetaClusterNamespaceKey(key string) (clusterName logicalcluster.Name, namespace, name string, err error) {
	invalidKey := fmt.Errorf("unexpected key format: %q", key)
	outerParts := strings.Split(key, "|")
	switch len(outerParts) {
	case 1:
		namespace, name, err := cache.SplitMetaNamespaceKey(outerParts[0])
		if err != nil {
			err = invalidKey
		}
		return "", namespace, name, err
	case 2:
		namespace, name, err := cache.SplitMetaNamespaceKey(outerParts[1])
		if err != nil {
			err = invalidKey
		}
		return logicalcluster.Name(outerParts[0]), namespace, name, err
	default:
		return "", "", "", invalidKey
	}
}
