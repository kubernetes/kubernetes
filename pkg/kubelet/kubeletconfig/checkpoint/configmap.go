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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	kubeletscheme "k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/scheme"
	utilcodec "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/codec"
)

const configMapConfigKey = "kubelet"

// configMapCheckpoint implements Checkpoint, backed by a v1/ConfigMap config source object
type configMapCheckpoint struct {
	kubeletCodecs *serializer.CodecFactory // codecs for the KubeletConfiguration
	configMap     *apiv1.ConfigMap
}

// NewConfigMapCheckpoint returns a Checkpoint backed by `cm`. `cm` must be non-nil
// and have a non-empty ObjectMeta.UID, or an error will be returned.
func NewConfigMapCheckpoint(cm *apiv1.ConfigMap) (Checkpoint, error) {
	if cm == nil {
		return nil, fmt.Errorf("ConfigMap must be non-nil to be treated as a Checkpoint")
	} else if len(cm.ObjectMeta.UID) == 0 {
		return nil, fmt.Errorf("ConfigMap must have a UID to be treated as a Checkpoint")
	}

	_, kubeletCodecs, err := kubeletscheme.NewSchemeAndCodecs()
	if err != nil {
		return nil, err
	}

	return &configMapCheckpoint{kubeletCodecs, cm}, nil
}

// UID returns the UID of a configMapCheckpoint
func (c *configMapCheckpoint) UID() string {
	return string(c.configMap.UID)
}

// Parse extracts the KubeletConfiguration from v1/ConfigMap checkpoints, applies defaults, and converts to the internal type
func (c *configMapCheckpoint) Parse() (*kubeletconfig.KubeletConfiguration, error) {
	const emptyCfgErr = "config was empty, but some parameters are required"

	if len(c.configMap.Data) == 0 {
		return nil, fmt.Errorf(emptyCfgErr)
	}

	// TODO(mtaufen): Once the KubeletConfiguration type is decomposed, extend this to a key for each sub-object
	config, ok := c.configMap.Data[configMapConfigKey]
	if !ok {
		return nil, fmt.Errorf("key %q not found in ConfigMap", configMapConfigKey)
	} else if len(config) == 0 {
		return nil, fmt.Errorf(emptyCfgErr)
	}

	return utilcodec.DecodeKubeletConfiguration(c.kubeletCodecs, []byte(config))
}

// Encode encodes a configMapCheckpoint
func (c *configMapCheckpoint) Encode() ([]byte, error) {
	cm := c.configMap
	encoder, err := utilcodec.NewJSONEncoder(apiv1.GroupName)
	if err != nil {
		return nil, err
	}
	data, err := runtime.Encode(encoder, cm)
	if err != nil {
		return nil, err
	}
	return data, nil
}

func (c *configMapCheckpoint) object() interface{} {
	return c.configMap
}
