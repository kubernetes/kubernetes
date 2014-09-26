/*
Copyright 2014 Google Inc. All rights reserved.

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

package minion

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
)

type CloudRegistry struct {
	cloud           cloudprovider.Interface
	matchRE         string
	staticResources *api.NodeResources
}

func NewCloudRegistry(cloud cloudprovider.Interface, matchRE string, staticResources *api.NodeResources) (*CloudRegistry, error) {
	return &CloudRegistry{
		cloud:           cloud,
		matchRE:         matchRE,
		staticResources: staticResources,
	}, nil
}

func (r *CloudRegistry) Contains(nodeID string) (bool, error) {
	instances, err := r.List()
	if err != nil {
		return false, err
	}
	for _, node := range instances.Items {
		if node.ID == nodeID {
			return true, nil
		}
	}
	return false, nil
}

func (r CloudRegistry) Delete(minion string) error {
	return fmt.Errorf("unsupported")
}

func (r CloudRegistry) Insert(minion string) error {
	return fmt.Errorf("unsupported")
}

func (r *CloudRegistry) List() (*api.MinionList, error) {
	instances, ok := r.cloud.Instances()
	if !ok {
		return nil, fmt.Errorf("cloud doesn't support instances")
	}
	matches, err := instances.List(r.matchRE)
	if err != nil {
		return nil, err
	}
	result := &api.MinionList{
		Items: make([]api.Minion, len(matches)),
	}
	for ix := range matches {
		result.Items[ix].ID = matches[ix]
		resources, err := instances.GetNodeResources(matches[ix])
		if err != nil {
			return nil, err
		}
		if resources == nil {
			resources = r.staticResources
		}
		if resources != nil {
			result.Items[ix].NodeResources = *resources
		}
	}
	return result, err
}
