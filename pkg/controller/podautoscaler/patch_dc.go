package podautoscaler

import (
	"k8s.io/apimachinery/pkg/runtime/schema"

	apimeta "k8s.io/apimachinery/pkg/api/meta"
)

func overrideMappingsForOapiDeploymentConfig(mappings []*apimeta.RESTMapping, err error, targetGK schema.GroupKind) ([]*apimeta.RESTMapping, error) {
	if (targetGK == schema.GroupKind{Kind: "DeploymentConfig"}) {
		err = nil
		mappings = []*apimeta.RESTMapping{
			{
				Resource:         schema.GroupVersionResource{Group: "apps.openshift.io", Version: "v1", Resource: "deploymentconfigs"},
				GroupVersionKind: schema.GroupVersionKind{Group: "apps.openshift.io", Version: "v1", Kind: "DeploymentConfig"},
			},
		}
	}
	return mappings, err
}
