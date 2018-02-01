/*
Copyright (c) 2017 GigaSpaces Technologies Ltd. All rights reserved

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

package cloudifyprovider

import (
	"fmt"
	cloudify "github.com/cloudify-incubator/cloudify-rest-go-client/cloudify"
	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

// Zones - struct with connection settings
type Zones struct {
	client *cloudify.Client
}

// GetZone is an implementation of Zones.GetZone
func (r *Zones) GetZone() (cloudprovider.Zone, error) {
	glog.V(4).Info("GetZone")
	return cloudprovider.Zone{
		FailureDomain: "FailureDomain",
		Region:        "Region",
	}, nil
}

// GetZoneByProviderID implements Zones.GetZoneByProviderID
// This is particularly useful in external cloud providers where the kubelet
// does not initialize node data.
func (r *Zones) GetZoneByProviderID(providerID string) (cloudprovider.Zone, error) {
	glog.Errorf("?GetZone: [%+v]", providerID)
	return cloudprovider.Zone{}, fmt.Errorf("GetZoneByProviderID not implemented")
}

// GetZoneByNodeName implements Zones.GetZoneByNodeName
// This is particularly useful in external cloud providers where the kubelet
// does not initialize node data.
func (r *Zones) GetZoneByNodeName(nodeName types.NodeName) (cloudprovider.Zone, error) {
	glog.Errorf("?GetZoneByNodeName: [%+v]", nodeName)
	return cloudprovider.Zone{}, fmt.Errorf("GetZoneByNodeName not imeplemented")
}

// NewZones - create instance with support kubernetes zones interface.
func NewZones(client *cloudify.Client) *Zones {
	return &Zones{
		client: client,
	}
}
