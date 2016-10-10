/*
Copyright 2016 The Kubernetes Authors.

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

package registrytest

import (
	"fmt"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/watch"
)

type ConfigMapRegistry struct {
	ConfigMap *api.ConfigMap
}

func (c *ConfigMapRegistry) ListConfigMaps(ctx api.Context, options *api.ListOptions) (*api.ConfigMapList, error) {
	return nil, fmt.Errorf("unimplemented!")
}

func (c *ConfigMapRegistry) WatchConfigMaps(ctx api.Context, options *api.ListOptions) (watch.Interface, error) {
	return nil, fmt.Errorf("unimplemented!")
}

func (c *ConfigMapRegistry) GetConfigMap(ctx api.Context, name string) (*api.ConfigMap, error) {
	if c.ConfigMap == nil {
		return nil, errors.NewNotFound(api.Resource("configmaps"), name)
	} else {
		return c.ConfigMap, nil
	}
}

func (c *ConfigMapRegistry) CreateConfigMap(ctx api.Context, cfg *api.ConfigMap) (*api.ConfigMap, error) {
	return nil, fmt.Errorf("unimplemented!")
}

func (c *ConfigMapRegistry) UpdateConfigMap(ctx api.Context, cfg *api.ConfigMap) (*api.ConfigMap, error) {
	return nil, fmt.Errorf("unimplemented!")
}

func (c *ConfigMapRegistry) DeleteConfigMap(ctx api.Context, name string) error {
	return fmt.Errorf("unimplemented!")
}
