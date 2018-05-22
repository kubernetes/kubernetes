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

package checkpoint

import (
	"fmt"

	apiv1 "k8s.io/api/core/v1"
)

const configMapConfigKey = "kubelet"

// configMapPayload implements Payload, backed by a v1/ConfigMap config source object
type configMapPayload struct {
	cm *apiv1.ConfigMap
}

var _ Payload = (*configMapPayload)(nil)

// NewConfigMapPayload constructs a Payload backed by a ConfigMap, which must have a non-empty UID
func NewConfigMapPayload(cm *apiv1.ConfigMap) (Payload, error) {
	if cm == nil {
		return nil, fmt.Errorf("ConfigMap must be non-nil")
	} else if cm.ObjectMeta.UID == "" {
		return nil, fmt.Errorf("ConfigMap must have a non-empty UID")
	} else if cm.ObjectMeta.ResourceVersion == "" {
		return nil, fmt.Errorf("ConfigMap must have a non-empty ResourceVersion")
	}

	return &configMapPayload{cm}, nil
}

func (p *configMapPayload) UID() string {
	return string(p.cm.UID)
}

func (p *configMapPayload) ResourceVersion() string {
	return p.cm.ResourceVersion
}

func (p *configMapPayload) Files() map[string]string {
	return p.cm.Data
}

func (p *configMapPayload) object() interface{} {
	return p.cm
}
