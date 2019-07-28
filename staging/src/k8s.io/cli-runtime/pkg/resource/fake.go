/*
Copyright 2017 The Kubernetes Authors.

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

package resource

import (
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/restmapper"
)

// FakeCategoryExpander is for testing only
var FakeCategoryExpander restmapper.CategoryExpander = restmapper.SimpleCategoryExpander{
	Expansions: map[string][]schema.GroupResource{
		"all": {
			{Group: "", Resource: "pods"},
			{Group: "", Resource: "replicationcontrollers"},
			{Group: "", Resource: "services"},
			{Group: "apps", Resource: "statefulsets"},
			{Group: "autoscaling", Resource: "horizontalpodautoscalers"},
			{Group: "batch", Resource: "jobs"},
			{Group: "batch", Resource: "cronjobs"},
			{Group: "extensions", Resource: "daemonsets"},
			{Group: "extensions", Resource: "deployments"},
			{Group: "extensions", Resource: "replicasets"},
		},
	},
}
