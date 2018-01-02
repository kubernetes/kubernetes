package testing

import (
	"testing"
	"time"

	"github.com/gophercloud/gophercloud/openstack/sharedfilesystems/v2/shares"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockCreateResponse(t)

	options := &shares.CreateOpts{Size: 1, Name: "my_test_share", ShareProto: "NFS"}
	n, err := shares.Create(client.ServiceClient(), options).Extract()

	th.AssertNoErr(t, err)
	th.AssertEquals(t, n.Name, "my_test_share")
	th.AssertEquals(t, n.Size, 1)
	th.AssertEquals(t, n.ShareProto, "NFS")
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockDeleteResponse(t)

	result := shares.Delete(client.ServiceClient(), shareID)
	th.AssertNoErr(t, result.Err)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockGetResponse(t)

	s, err := shares.Get(client.ServiceClient(), shareID).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, s, &shares.Share{
		AvailabilityZone:   "nova",
		ShareNetworkID:     "713df749-aac0-4a54-af52-10f6c991e80c",
		ShareServerID:      "e268f4aa-d571-43dd-9ab3-f49ad06ffaef",
		SnapshotID:         "",
		ID:                 shareID,
		Size:               1,
		ShareType:          "25747776-08e5-494f-ab40-a64b9d20d8f7",
		ShareTypeName:      "default",
		ConsistencyGroupID: "9397c191-8427-4661-a2e8-b23820dc01d4",
		ProjectID:          "16e1ab15c35a457e9c2b2aa189f544e1",
		Metadata: map[string]string{
			"project": "my_app",
			"aim":     "doc",
		},
		Status:                   "available",
		Description:              "My custom share London",
		Host:                     "manila2@generic1#GENERIC1",
		HasReplicas:              false,
		ReplicationType:          "",
		TaskState:                "",
		SnapshotSupport:          true,
		Name:                     "my_test_share",
		CreatedAt:                time.Date(2015, time.September, 18, 10, 25, 24, 0, time.UTC),
		ShareProto:               "NFS",
		VolumeType:               "default",
		SourceCgsnapshotMemberID: "",
		IsPublic:                 true,
		Links: []map[string]string{
			{
				"href": "http://172.18.198.54:8786/v2/16e1ab15c35a457e9c2b2aa189f544e1/shares/011d21e2-fbc3-4e4a-9993-9ea223f73264",
				"rel":  "self",
			},
			{
				"href": "http://172.18.198.54:8786/16e1ab15c35a457e9c2b2aa189f544e1/shares/011d21e2-fbc3-4e4a-9993-9ea223f73264",
				"rel":  "bookmark",
			},
		},
	})
}

func TestGetExportLocationsSuccess(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockGetExportLocationsResponse(t)

	c := client.ServiceClient()
	// Client c must have Microversion set; minimum supported microversion for Get Export Locations is 2.14
	c.Microversion = "2.14"

	s, err := shares.GetExportLocations(c, shareID).Extract()

	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, s, []shares.ExportLocation{
		{
			Path:            "127.0.0.1:/var/lib/manila/mnt/share-9a922036-ad26-4d27-b955-7a1e285fa74d",
			ShareInstanceID: "011d21e2-fbc3-4e4a-9993-9ea223f73264",
			IsAdminOnly:     false,
			ID:              "80ed63fc-83bc-4afc-b881-da4a345ac83d",
			Preferred:       false,
		},
	})
}

func TestGrantAcessSuccess(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockGrantAccessResponse(t)

	c := client.ServiceClient()
	// Client c must have Microversion set; minimum supported microversion for Grant Access is 2.7
	c.Microversion = "2.7"

	var grantAccessReq shares.GrantAccessOpts
	grantAccessReq.AccessType = "ip"
	grantAccessReq.AccessTo = "0.0.0.0/0"
	grantAccessReq.AccessLevel = "rw"

	s, err := shares.GrantAccess(c, shareID, grantAccessReq).Extract()

	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, s, &shares.AccessRight{
		ShareID:     "011d21e2-fbc3-4e4a-9993-9ea223f73264",
		AccessType:  "ip",
		AccessTo:    "0.0.0.0/0",
		AccessKey:   "",
		AccessLevel: "rw",
		State:       "new",
		ID:          "a2f226a5-cee8-430b-8a03-78a59bd84ee8",
	})
}
