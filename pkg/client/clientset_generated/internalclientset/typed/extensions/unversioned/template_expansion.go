/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package unversioned

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

// The TemplateExpansion interface allows manually adding extra methods to the TemplateInterface.
type TemplateExpansion interface {
	Process(tp *extensions.TemplateParameters) (*api.List, error)
}

func (t *templates) Process(tp *extensions.TemplateParameters) (*api.List, error) {
	result := &api.List{}
	err := t.client.Put().Namespace(t.ns).
		Resource("templates").
		Name(tp.Name).
		SubResource("process").
		Body(tp).Do().Into(result)

	// Manually fixup missing/extra fields for export so that we can
	// pipe the result back into `kubectl -f -` to instantiate the values
	// from the expanded Template.
	result.Kind = "List"
	result.TypeMeta.APIVersion = "v1"
	result.ListMeta.SelfLink = ""
	return result, err
}
