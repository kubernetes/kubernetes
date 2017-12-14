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

package kubectl

import (
	"bytes"
	"fmt"

	"k8s.io/api/core/v1"
	api "k8s.io/kubernetes/pkg/apis/core"
	apiv1 "k8s.io/kubernetes/pkg/apis/core/v1"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
)

func printPodTemplate(template *v1.PodTemplateSpec) (string, error) {
	buf := bytes.NewBuffer([]byte{})
	internalTemplate := &api.PodTemplateSpec{}
	if err := apiv1.Convert_v1_PodTemplateSpec_To_core_PodTemplateSpec(template, internalTemplate, nil); err != nil {
		return "", fmt.Errorf("failed to convert podtemplate, %v", err)
	}
	w := printersinternal.NewPrefixWriter(buf)
	printersinternal.DescribePodTemplate(internalTemplate, w)
	return buf.String(), nil
}
