package apiserver

import (
	"fmt"
	"strings"

	"github.com/kcp-dev/logicalcluster/v3"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/storage/storagebackend"
)

type apiBindingAwareCRDRESTOptionsGetter struct {
	delegate generic.RESTOptionsGetter
	crd      *apiextensionsv1.CustomResourceDefinition
}

func (t apiBindingAwareCRDRESTOptionsGetter) GetRESTOptions(resource schema.GroupResource, obj runtime.Object) (generic.RESTOptions, error) {
	ret, err := t.delegate.GetRESTOptions(resource, obj)
	if err != nil {
		return ret, err
	}

	// assign some KCP metadata that are used by the reflector from the watch cache
	ret.StorageConfig.KcpExtraStorageMetadata = &storagebackend.KcpStorageMetadata{IsCRD: true}

	// Priority 1: wildcard partial metadata requests. These have been assigned a fake UID that ends with
	// .wildcard.partial-metadata. If this is present, we don't want to modify the ResourcePrefix, which means that
	// a wildcard partial metadata list/watch request will return every CR from every CRD for that group-resource, which
	// could include instances from normal CRDs as well as those coming from CRDs with different identities. This would
	// return e.g. everything under
	//
	//   - /registry/mygroup.io/widgets/customresources/...
	//   - /registry/mygroup.io/widgets/identity1234/...
	//   - /registry/mygroup.io/widgets/identity4567/...
	if strings.HasSuffix(string(t.crd.UID), ".wildcard.partial-metadata") {
		ret.StorageConfig.KcpExtraStorageMetadata.Cluster = genericapirequest.Cluster{Wildcard: true, PartialMetadataRequest: true}
		return ret, nil
	}

	ret.StorageConfig.KcpExtraStorageMetadata.Cluster.Wildcard = true

	// Normal CRDs (not coming from an APIBinding) are stored in e.g. /registry/mygroup.io/widgets/<cluster name>/...
	if _, bound := t.crd.Annotations["apis.kcp.io/bound-crd"]; !bound {
		ret.ResourcePrefix += "/customresources"

		clusterName := logicalcluster.From(t.crd)
		if clusterName != "system:system-crds" {
			// For all normal CRDs outside of the system:system-crds logical cluster, tell the watch cache the name
			// of the logical cluster to use, and turn off wildcarding. This ensures the watch cache is just for
			// this logical cluster.
			ret.StorageConfig.KcpExtraStorageMetadata.Cluster.Name = clusterName
			ret.StorageConfig.KcpExtraStorageMetadata.Cluster.Wildcard = false
		}
		return ret, nil
	}

	// Bound CRDs must have the associated identity annotation
	apiIdentity := t.crd.Annotations["apis.kcp.io/identity"]
	if apiIdentity == "" {
		return generic.RESTOptions{}, fmt.Errorf("missing 'apis.kcp.io/identity' annotation on CRD %s|%s for %s.%s", logicalcluster.From(t.crd), t.crd.Name, t.crd.Spec.Names.Plural, t.crd.Spec.Group)
	}

	// Modify the ResourcePrefix so it results in e.g. /registry/mygroup.io/widgets/identity4567/...
	ret.ResourcePrefix += "/" + apiIdentity

	return ret, err
}
