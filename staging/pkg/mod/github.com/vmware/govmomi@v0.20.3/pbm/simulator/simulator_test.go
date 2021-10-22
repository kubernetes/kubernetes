/*
Copyright (c) 2018 VMware, Inc. All Rights Reserved.

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

package simulator

import (
	"context"
	"log"
	"reflect"
	"sort"
	"testing"

	"github.com/vmware/govmomi"
	"github.com/vmware/govmomi/pbm"
	"github.com/vmware/govmomi/pbm/types"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/simulator"
	"github.com/vmware/govmomi/view"
	"github.com/vmware/govmomi/vim25/mo"
	vim "github.com/vmware/govmomi/vim25/types"
)

// TestSimulator is a copy of pbm/client_test.go:ClientTest
// The pbm package cannot import the pbm/simulator package due to cyclic dependency.
func TestSimulator(t *testing.T) {
	ctx := context.Background()

	model := simulator.VPX()

	defer model.Remove()
	err := model.Create()
	if err != nil {
		log.Fatal(err)
	}

	s := model.Service.NewServer()
	defer s.Close()

	model.Service.RegisterSDK(New())

	c, err := govmomi.NewClient(ctx, s.URL, true)
	if err != nil {
		t.Fatal(err)
	}

	pc, err := pbm.NewClient(ctx, c.Client)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("PBM version=%s", pc.ServiceContent.AboutInfo.Version)

	rtype := types.PbmProfileResourceType{
		ResourceType: string(types.PbmProfileResourceTypeEnumSTORAGE),
	}

	category := types.PbmProfileCategoryEnumREQUIREMENT

	// 1. Query all the profiles on the vCenter.
	ids, err := pc.QueryProfile(ctx, rtype, string(category))
	if err != nil {
		t.Fatal(err)
	}

	var qids []string

	for _, id := range ids {
		qids = append(qids, id.UniqueId)
	}

	var cids []string

	// 2. Retrieve the content of all profiles.
	policies, err := pc.RetrieveContent(ctx, ids)
	if err != nil {
		t.Fatal(err)
	}

	for i := range policies {
		profile := policies[i].GetPbmProfile()
		cids = append(cids, profile.ProfileId.UniqueId)
	}

	sort.Strings(qids)
	sort.Strings(cids)

	// Check whether ids retreived from QueryProfile and RetrieveContent are identical.
	if !reflect.DeepEqual(qids, cids) {
		t.Error("ids mismatch")
	}

	// 3. Get list of datastores in a cluster if cluster name is specified.
	root := c.ServiceContent.RootFolder
	var datastores []vim.ManagedObjectReference
	var kind []string
	clusterName := "DC0_C0"
	if clusterName == "" {
		kind = []string{"Datastore"}
	} else {
		kind = []string{"ClusterComputeResource"}
	}

	m := view.NewManager(c.Client)

	v, err := m.CreateContainerView(ctx, root, kind, true)
	if err != nil {
		t.Fatal(err)
	}

	if clusterName == "" {
		datastores, err = v.Find(ctx, kind, nil)
		if err != nil {
			t.Fatal(err)
		}
	} else {
		var cluster mo.ClusterComputeResource

		err = v.RetrieveWithFilter(ctx, kind, []string{"datastore"}, &cluster, property.Filter{"name": clusterName})
		if err != nil {
			t.Fatal(err)
		}

		datastores = cluster.Datastore
	}

	_ = v.Destroy(ctx)

	t.Logf("checking %d datatores for compatibility results", len(datastores))

	var hubs []types.PbmPlacementHub

	for _, ds := range datastores {
		hubs = append(hubs, types.PbmPlacementHub{
			HubType: ds.Type,
			HubId:   ds.Value,
		})
	}

	var req []types.BasePbmPlacementRequirement

	for _, id := range ids {
		req = append(req, &types.PbmPlacementCapabilityProfileRequirement{
			ProfileId: id,
		})
	}

	// 4. Get the compatibility results for all the profiles on the vCenter.
	res, err := pc.CheckRequirements(ctx, hubs, nil, req)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("CheckRequirements results: %d", len(res))

	// user spec for the profile.
	// VSAN profile with 2 capability instances - hostFailuresToTolerate = 2, stripeWidth = 1
	pbmCreateSpecForVSAN := pbm.CapabilityProfileCreateSpec{
		Name:        "Kubernetes-VSAN-TestPolicy",
		Description: "VSAN Test policy create",
		Category:    string(types.PbmProfileCategoryEnumREQUIREMENT),
		CapabilityList: []pbm.Capability{
			pbm.Capability{
				ID:        "hostFailuresToTolerate",
				Namespace: "VSAN",
				PropertyList: []pbm.Property{
					pbm.Property{
						ID:       "hostFailuresToTolerate",
						Value:    "2",
						DataType: "int",
					},
				},
			},
			pbm.Capability{
				ID:        "stripeWidth",
				Namespace: "VSAN",
				PropertyList: []pbm.Property{
					pbm.Property{
						ID:       "stripeWidth",
						Value:    "1",
						DataType: "int",
					},
				},
			},
		},
	}

	// Create PBM capability spec for the above defined user spec.
	createSpecVSAN, err := pbm.CreateCapabilityProfileSpec(pbmCreateSpecForVSAN)
	if err != nil {
		t.Fatal(err)
	}

	// 5. Create SPBM VSAN profile.
	vsanProfileID, err := pc.CreateProfile(ctx, *createSpecVSAN)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("VSAN Profile: %q successfully created", vsanProfileID.UniqueId)

	// 6. Verify if profile created exists by issuing a RetrieveContent request.
	_, err = pc.RetrieveContent(ctx, []types.PbmProfileId{*vsanProfileID})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Profile: %q exists on vCenter", vsanProfileID.UniqueId)

	// 7. Get compatible datastores for the VSAN profile.
	compatibleDatastores := res.CompatibleDatastores()
	t.Logf("Found %d compatible-datastores for profile: %q", len(compatibleDatastores), vsanProfileID.UniqueId)

	// 8. Get non-compatible datastores for the VSAN profile.
	nonCompatibleDatastores := res.NonCompatibleDatastores()
	t.Logf("Found %d non-compatible datastores for profile: %q", len(nonCompatibleDatastores), vsanProfileID.UniqueId)

	// Check whether count of compatible and non-compatible datastores match the total number of datastores.
	if (len(nonCompatibleDatastores) + len(compatibleDatastores)) != len(datastores) {
		t.Error("datastore count mismatch")
	}

	// user spec for the profile.
	// VSAN profile with 2 capability instances - stripeWidth = 1 and an SIOC profile.
	pbmCreateSpecVSANandSIOC := pbm.CapabilityProfileCreateSpec{
		Name:        "Kubernetes-VSAN-SIOC-TestPolicy",
		Description: "VSAN-SIOC-Test policy create",
		Category:    string(types.PbmProfileCategoryEnumREQUIREMENT),
		CapabilityList: []pbm.Capability{
			pbm.Capability{
				ID:        "stripeWidth",
				Namespace: "VSAN",
				PropertyList: []pbm.Property{
					pbm.Property{
						ID:       "stripeWidth",
						Value:    "1",
						DataType: "int",
					},
				},
			},
			pbm.Capability{
				ID:        "spm@DATASTOREIOCONTROL",
				Namespace: "spm",
				PropertyList: []pbm.Property{
					pbm.Property{
						ID:       "limit",
						Value:    "200",
						DataType: "int",
					},
					pbm.Property{
						ID:       "reservation",
						Value:    "1000",
						DataType: "int",
					},
					pbm.Property{
						ID:       "shares",
						Value:    "2000",
						DataType: "int",
					},
				},
			},
		},
	}

	// Create PBM capability spec for the above defined user spec.
	createSpecVSANandSIOC, err := pbm.CreateCapabilityProfileSpec(pbmCreateSpecVSANandSIOC)
	if err != nil {
		t.Fatal(err)
	}

	// 9. Create SPBM VSAN profile.
	vsansiocProfileID, err := pc.CreateProfile(ctx, *createSpecVSANandSIOC)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("VSAN-SIOC Profile: %q successfully created", vsansiocProfileID.UniqueId)

	// 9. Get ProfileID by Name
	profileID, err := pc.ProfileIDByName(ctx, "Kubernetes-VSAN-SIOC-TestPolicy")
	if err != nil {
		t.Fatal(err)
	}

	if vsansiocProfileID.UniqueId != profileID {
		t.Errorf("vsan-sioc profile: %q and retrieved profileID: %q successfully matched", vsansiocProfileID.UniqueId, profileID)
	}
	t.Logf("VSAN-SIOC profile: %q and retrieved profileID: %q successfully matched", vsansiocProfileID.UniqueId, profileID)

	// 10. Delete VSAN and VSAN-SIOC profile.
	_, err = pc.DeleteProfile(ctx, []types.PbmProfileId{*vsanProfileID, *vsansiocProfileID})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Profile: %+v successfully deleted", []types.PbmProfileId{*vsanProfileID, *vsansiocProfileID})
}
