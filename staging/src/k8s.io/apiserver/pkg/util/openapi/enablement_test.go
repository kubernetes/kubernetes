/*
Copyright 2021 The Kubernetes Authors.

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

package openapi

import (
	"fmt"
	"strings"
	"testing"

	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

func schema_k8sio_api_apps_v1_DeploymentCondition(ref common.ReferenceCallback) common.OpenAPIDefinition {
	return common.OpenAPIDefinition{
		Schema: spec.Schema{
			SchemaProps: spec.SchemaProps{
				Description: "DeploymentCondition describes the state of a deployment at a certain point.",
				Type:        []string{"object"},
				Properties: map[string]spec.Schema{
					"type": {
						SchemaProps: spec.SchemaProps{
							Description: "Type of deployment condition.\n\nPossible enum values:\n - `Available`: Available means the deployment is available, ie. at least the minimum available replicas required are up and running for at least minReadySeconds.\n - `Progressing`: Progressing means the deployment is progressing. Progress for a deployment is considered when a new replica set is created or adopted, and when new pods scale up or old pods scale down. Progress is not estimated for paused deployments or when progressDeadlineSeconds is not specified.\n - `ReplicaFailure`: ReplicaFailure is added in a deployment when one of its pods fails to be created or deleted.",
							Default:     "",
							Type:        []string{"string"},
							Format:      "",
							Enum:        []interface{}{"Available", "Progressing", "ReplicaFailure"}},
					},
					"status": {
						SchemaProps: spec.SchemaProps{
							Description: "Status of the condition, one of True, False, Unknown.\n\nPossible enum values:\n - `False`:\n - `True`:\n - `Unknown`:",
							Default:     "",
							Type:        []string{"string"},
							Format:      "",
							Enum:        []interface{}{"False", "True", "Unknown"}},
					},
					"lastUpdateTime": {
						SchemaProps: spec.SchemaProps{
							Description: "The last time this condition was updated.",
							Default:     map[string]interface{}{},
							Ref:         ref("k8s.io/apimachinery/pkg/apis/meta/v1.Time"),
						},
					},
					"lastTransitionTime": {
						SchemaProps: spec.SchemaProps{
							Description: "Last time the condition transitioned from one status to another.",
							Default:     map[string]interface{}{},
							Ref:         ref("k8s.io/apimachinery/pkg/apis/meta/v1.Time"),
						},
					},
					"reason": {
						SchemaProps: spec.SchemaProps{
							Description: "The reason for the condition's last transition.",
							Type:        []string{"string"},
							Format:      "",
						},
					},
					"message": {
						SchemaProps: spec.SchemaProps{
							Description: "A human readable message indicating details about the transition.",
							Type:        []string{"string"},
							Format:      "",
						},
					},
				},
				Required: []string{"type", "status"},
			},
		},
		Dependencies: []string{
			"k8s.io/apimachinery/pkg/apis/meta/v1.Time"},
	}
}

var getOpenAPIDefs common.GetOpenAPIDefinitions = func(ref common.ReferenceCallback) map[string]common.OpenAPIDefinition {
	return map[string]common.OpenAPIDefinition{
		"k8s.io/api/apps/v1.DeploymentCondition": schema_k8sio_api_apps_v1_DeploymentCondition(ref),
	}
}

func TestGetOpenAPIDefinitionsWithoutDisabledFeatures(t *testing.T) {
	for _, tc := range []struct {
		enabled        bool
		shouldHaveEnum bool
	}{
		{
			enabled:        true,
			shouldHaveEnum: true,
		},
		{
			enabled:        false,
			shouldHaveEnum: false,
		},
	} {
		t.Run(fmt.Sprintf("enabled=%v", tc.enabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.OpenAPIEnums, tc.enabled)
			defs := GetOpenAPIDefinitionsWithoutDisabledFeatures(getOpenAPIDefs)(func(path string) spec.Ref {
				return spec.Ref{}
			})
			def := defs["k8s.io/api/apps/v1.DeploymentCondition"]
			enumAppeared := false
			for _, prop := range def.Schema.Properties {
				if strings.Contains(prop.Description, "enum") {
					enumAppeared = true
					if !tc.shouldHaveEnum {
						t.Errorf("enum appeared, description: %s", prop.Description)
					}
				}
				if len(prop.Enum) != 0 {
					enumAppeared = true
					if !tc.shouldHaveEnum {
						t.Errorf("enum appeared, enum: %v", prop.Enum)
					}
				}
			}
			if !enumAppeared && tc.shouldHaveEnum {
				t.Errorf("enum did not appear")
			}
		})
	}

}
