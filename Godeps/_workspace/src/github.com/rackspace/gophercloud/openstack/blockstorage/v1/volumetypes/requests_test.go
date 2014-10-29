package volumetypes

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	"github.com/rackspace/gophercloud/testhelper/client"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockListResponse(t)

	count := 0

	List(client.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractVolumeTypes(page)
		if err != nil {
			t.Errorf("Failed to extract volume types: %v", err)
			return false, err
		}

		expected := []VolumeType{
			VolumeType{
				ID:   "289da7f8-6440-407c-9fb4-7db01ec49164",
				Name: "vol-type-001",
				ExtraSpecs: map[string]interface{}{
					"capabilities": "gpu",
				},
			},
			VolumeType{
				ID:         "96c3bda7-c82a-4f50-be73-ca7621794835",
				Name:       "vol-type-002",
				ExtraSpecs: map[string]interface{}{},
			},
		}

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})

	if count != 1 {
		t.Errorf("Expected 1 page, got %d", count)
	}
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockGetResponse(t)

	vt, err := Get(client.ServiceClient(), "d32019d3-bc6e-4319-9c1d-6722fc136a22").Extract()
	th.AssertNoErr(t, err)

	th.AssertDeepEquals(t, vt.ExtraSpecs, map[string]interface{}{"serverNumber": "2"})
	th.AssertEquals(t, vt.Name, "vol-type-001")
	th.AssertEquals(t, vt.ID, "d32019d3-bc6e-4319-9c1d-6722fc136a22")
}

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/types", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
    "volume_type": {
        "name": "vol-type-001"
    }
}
			`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, `
{
    "volume_type": {
        "name": "vol-type-001",
        "id": "d32019d3-bc6e-4319-9c1d-6722fc136a22"
    }
}
		`)
	})

	options := &CreateOpts{Name: "vol-type-001"}
	n, err := Create(client.ServiceClient(), options).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, n.Name, "vol-type-001")
	th.AssertEquals(t, n.ID, "d32019d3-bc6e-4319-9c1d-6722fc136a22")
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/types/d32019d3-bc6e-4319-9c1d-6722fc136a22", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		w.WriteHeader(http.StatusAccepted)
	})

	err := Delete(client.ServiceClient(), "d32019d3-bc6e-4319-9c1d-6722fc136a22").ExtractErr()
	th.AssertNoErr(t, err)
}
