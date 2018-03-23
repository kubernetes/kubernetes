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

package vclib

import (
	"context"
	"fmt"

	"github.com/golang/glog"
	"github.com/vmware/govmomi/pbm"

	pbmtypes "github.com/vmware/govmomi/pbm/types"
	"github.com/vmware/govmomi/vim25"
)

// PbmClient is extending govmomi pbm, and provides functions to get compatible list of datastore for given policy
type PbmClient struct {
	*pbm.Client
}

// NewPbmClient returns a new PBM Client object
func NewPbmClient(ctx context.Context, client *vim25.Client) (*PbmClient, error) {
	pbmClient, err := pbm.NewClient(ctx, client)
	if err != nil {
		glog.Errorf("Failed to create new Pbm Client. err: %+v", err)
		return nil, err
	}
	return &PbmClient{pbmClient}, nil
}

// IsDatastoreCompatible check if the datastores is compatible for given storage policy id
// if datastore is not compatible with policy, fault message with the Datastore Name is returned
func (pbmClient *PbmClient) IsDatastoreCompatible(ctx context.Context, storagePolicyID string, datastore *Datastore) (bool, string, error) {
	faultMessage := ""
	placementHub := pbmtypes.PbmPlacementHub{
		HubType: datastore.Reference().Type,
		HubId:   datastore.Reference().Value,
	}
	hubs := []pbmtypes.PbmPlacementHub{placementHub}
	req := []pbmtypes.BasePbmPlacementRequirement{
		&pbmtypes.PbmPlacementCapabilityProfileRequirement{
			ProfileId: pbmtypes.PbmProfileId{
				UniqueId: storagePolicyID,
			},
		},
	}
	compatibilityResult, err := pbmClient.CheckRequirements(ctx, hubs, nil, req)
	if err != nil {
		glog.Errorf("Error occurred for CheckRequirements call. err %+v", err)
		return false, "", err
	}
	if compatibilityResult != nil && len(compatibilityResult) > 0 {
		compatibleHubs := compatibilityResult.CompatibleDatastores()
		if compatibleHubs != nil && len(compatibleHubs) > 0 {
			return true, "", nil
		}
		dsName, err := datastore.ObjectName(ctx)
		if err != nil {
			glog.Errorf("Failed to get datastore ObjectName")
			return false, "", err
		}
		if compatibilityResult[0].Error[0].LocalizedMessage == "" {
			faultMessage = "Datastore: " + dsName + " is not compatible with the storage policy."
		} else {
			faultMessage = "Datastore: " + dsName + " is not compatible with the storage policy. LocalizedMessage: " + compatibilityResult[0].Error[0].LocalizedMessage + "\n"
		}
		return false, faultMessage, nil
	}
	return false, "", fmt.Errorf("compatibilityResult is nil or empty")
}

// GetCompatibleDatastores filters and returns compatible list of datastores for given storage policy id
// For Non Compatible Datastores, fault message with the Datastore Name is also returned
func (pbmClient *PbmClient) GetCompatibleDatastores(ctx context.Context, dc *Datacenter, storagePolicyID string, datastores []*DatastoreInfo) ([]*DatastoreInfo, string, error) {
	var (
		dsMorNameMap                                = getDsMorNameMap(ctx, datastores)
		localizedMessagesForNotCompatibleDatastores = ""
	)
	compatibilityResult, err := pbmClient.GetPlacementCompatibilityResult(ctx, storagePolicyID, datastores)
	if err != nil {
		glog.Errorf("Error occurred while retrieving placement compatibility result for datastores: %+v with storagePolicyID: %s. err: %+v", datastores, storagePolicyID, err)
		return nil, "", err
	}
	compatibleHubs := compatibilityResult.CompatibleDatastores()
	var compatibleDatastoreList []*DatastoreInfo
	for _, hub := range compatibleHubs {
		compatibleDatastoreList = append(compatibleDatastoreList, getDatastoreFromPlacementHub(datastores, hub))
	}
	for _, res := range compatibilityResult {
		for _, err := range res.Error {
			dsName := dsMorNameMap[res.Hub.HubId]
			localizedMessage := ""
			if err.LocalizedMessage != "" {
				localizedMessage = "Datastore: " + dsName + " not compatible with the storage policy. LocalizedMessage: " + err.LocalizedMessage + "\n"
			} else {
				localizedMessage = "Datastore: " + dsName + " not compatible with the storage policy. \n"
			}
			localizedMessagesForNotCompatibleDatastores += localizedMessage
		}
	}
	// Return an error if there are no compatible datastores.
	if len(compatibleHubs) < 1 {
		glog.Errorf("No compatible datastores found that satisfy the storage policy requirements: %s", storagePolicyID)
		return nil, localizedMessagesForNotCompatibleDatastores, fmt.Errorf("No compatible datastores found that satisfy the storage policy requirements")
	}
	return compatibleDatastoreList, localizedMessagesForNotCompatibleDatastores, nil
}

// GetPlacementCompatibilityResult gets placement compatibility result based on storage policy requirements.
func (pbmClient *PbmClient) GetPlacementCompatibilityResult(ctx context.Context, storagePolicyID string, datastore []*DatastoreInfo) (pbm.PlacementCompatibilityResult, error) {
	var hubs []pbmtypes.PbmPlacementHub
	for _, ds := range datastore {
		hubs = append(hubs, pbmtypes.PbmPlacementHub{
			HubType: ds.Reference().Type,
			HubId:   ds.Reference().Value,
		})
	}
	req := []pbmtypes.BasePbmPlacementRequirement{
		&pbmtypes.PbmPlacementCapabilityProfileRequirement{
			ProfileId: pbmtypes.PbmProfileId{
				UniqueId: storagePolicyID,
			},
		},
	}
	res, err := pbmClient.CheckRequirements(ctx, hubs, nil, req)
	if err != nil {
		glog.Errorf("Error occurred for CheckRequirements call. err: %+v", err)
		return nil, err
	}
	return res, nil
}

// getDataStoreForPlacementHub returns matching datastore associated with given pbmPlacementHub
func getDatastoreFromPlacementHub(datastore []*DatastoreInfo, pbmPlacementHub pbmtypes.PbmPlacementHub) *DatastoreInfo {
	for _, ds := range datastore {
		if ds.Reference().Type == pbmPlacementHub.HubType && ds.Reference().Value == pbmPlacementHub.HubId {
			return ds
		}
	}
	return nil
}

// getDsMorNameMap returns map of ds Mor and Datastore Object Name
func getDsMorNameMap(ctx context.Context, datastores []*DatastoreInfo) map[string]string {
	dsMorNameMap := make(map[string]string)
	for _, ds := range datastores {
		dsObjectName, err := ds.ObjectName(ctx)
		if err == nil {
			dsMorNameMap[ds.Reference().Value] = dsObjectName
		} else {
			glog.Errorf("Error occurred while getting datastore object name. err: %+v", err)
		}
	}
	return dsMorNameMap
}
