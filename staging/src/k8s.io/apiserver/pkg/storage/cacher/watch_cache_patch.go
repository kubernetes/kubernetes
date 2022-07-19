package cacher

import (
	"context"

	"github.com/kcp-dev/logicalcluster/v2"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/klog/v2"
)

// clusterNameGetter as of today it satisfies unstructured.Unstructured since it implements metav1.Object directly
type clusterNameGetter interface {
	GetZZZ_DeprecatedClusterName() string
}

// createClusterAwareContext extracts the clusterName from the given object and puts it into a context
// the context is used by the key function to compute the key under which the object will be stored
//
// background:
//
// resources in the db are stored without the clusterName, since the reflector used by the cache uses the logicalcluster.Wildcard
// the clusterName will be assigned to object by the storage layer upon retrieval.
// we need take it into consideration and change the key to contain the clusterName
// because this is how clients are going to be retrieving data from the cache.
func createClusterAwareContext(object runtime.Object) context.Context {
	var clusterName string

	switch t := object.(type) {
	case metav1.ObjectMetaAccessor:
		clusterName = t.GetObjectMeta().GetZZZ_DeprecatedClusterName()
	case clusterNameGetter:
		clusterName = t.GetZZZ_DeprecatedClusterName()
	default:
		klog.Warningf("unknown object, could not get a clusterName and a namespace from: %T", object)
		return context.Background()
	}

	return genericapirequest.WithCluster(context.Background(), genericapirequest.Cluster{Name: logicalcluster.New(clusterName)})
}
