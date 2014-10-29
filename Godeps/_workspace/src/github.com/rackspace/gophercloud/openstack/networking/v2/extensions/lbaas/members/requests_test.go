package members

import (
	"fmt"
	"net/http"
	"testing"

	fake "github.com/rackspace/gophercloud/openstack/networking/v2/common"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestURLs(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.AssertEquals(t, th.Endpoint()+"v2.0/lb/members", rootURL(fake.ServiceClient()))
}

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/lb/members", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
   "members":[
      {
         "status":"ACTIVE",
         "weight":1,
         "admin_state_up":true,
         "tenant_id":"83657cfcdfe44cd5920adaf26c48ceea",
         "pool_id":"72741b06-df4d-4715-b142-276b6bce75ab",
         "address":"10.0.0.4",
         "protocol_port":80,
         "id":"701b531b-111a-4f21-ad85-4795b7b12af6"
      },
      {
         "status":"ACTIVE",
         "weight":1,
         "admin_state_up":true,
         "tenant_id":"83657cfcdfe44cd5920adaf26c48ceea",
         "pool_id":"72741b06-df4d-4715-b142-276b6bce75ab",
         "address":"10.0.0.3",
         "protocol_port":80,
         "id":"beb53b4d-230b-4abd-8118-575b8fa006ef"
      }
   ]
}
      `)
	})

	count := 0

	List(fake.ServiceClient(), ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractMembers(page)
		if err != nil {
			t.Errorf("Failed to extract members: %v", err)
			return false, err
		}

		expected := []Member{
			Member{
				Status:       "ACTIVE",
				Weight:       1,
				AdminStateUp: true,
				TenantID:     "83657cfcdfe44cd5920adaf26c48ceea",
				PoolID:       "72741b06-df4d-4715-b142-276b6bce75ab",
				Address:      "10.0.0.4",
				ProtocolPort: 80,
				ID:           "701b531b-111a-4f21-ad85-4795b7b12af6",
			},
			Member{
				Status:       "ACTIVE",
				Weight:       1,
				AdminStateUp: true,
				TenantID:     "83657cfcdfe44cd5920adaf26c48ceea",
				PoolID:       "72741b06-df4d-4715-b142-276b6bce75ab",
				Address:      "10.0.0.3",
				ProtocolPort: 80,
				ID:           "beb53b4d-230b-4abd-8118-575b8fa006ef",
			},
		}

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})

	if count != 1 {
		t.Errorf("Expected 1 page, got %d", count)
	}
}

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/lb/members", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
  "member": {
    "tenant_id": "453105b9-1754-413f-aab1-55f1af620750",
		"pool_id": "foo",
    "address": "192.0.2.14",
    "protocol_port":8080
  }
}
      `)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, `
{
  "member": {
    "id": "975592ca-e308-48ad-8298-731935ee9f45",
    "address": "192.0.2.14",
    "protocol_port": 8080,
    "tenant_id": "453105b9-1754-413f-aab1-55f1af620750",
    "admin_state_up":true,
    "weight": 1,
    "status": "DOWN"
  }
}
    `)
	})

	options := CreateOpts{
		TenantID:     "453105b9-1754-413f-aab1-55f1af620750",
		Address:      "192.0.2.14",
		ProtocolPort: 8080,
		PoolID:       "foo",
	}
	_, err := Create(fake.ServiceClient(), options).Extract()
	th.AssertNoErr(t, err)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/lb/members/975592ca-e308-48ad-8298-731935ee9f45", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
   "member":{
      "id":"975592ca-e308-48ad-8298-731935ee9f45",
      "address":"192.0.2.14",
      "protocol_port":8080,
      "tenant_id":"453105b9-1754-413f-aab1-55f1af620750",
      "admin_state_up":true,
      "weight":1,
      "status":"DOWN"
   }
}
      `)
	})

	m, err := Get(fake.ServiceClient(), "975592ca-e308-48ad-8298-731935ee9f45").Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, "975592ca-e308-48ad-8298-731935ee9f45", m.ID)
	th.AssertEquals(t, "192.0.2.14", m.Address)
	th.AssertEquals(t, 8080, m.ProtocolPort)
	th.AssertEquals(t, "453105b9-1754-413f-aab1-55f1af620750", m.TenantID)
	th.AssertEquals(t, true, m.AdminStateUp)
	th.AssertEquals(t, 1, m.Weight)
	th.AssertEquals(t, "DOWN", m.Status)
}

func TestUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/lb/members/332abe93-f488-41ba-870b-2ac66be7f853", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
   "member":{
      "admin_state_up":false
   }
}
      `)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
   "member":{
      "status":"PENDING_UPDATE",
      "protocol_port":8080,
      "weight":1,
      "admin_state_up":false,
      "tenant_id":"4fd44f30292945e481c7b8a0c8908869",
      "pool_id":"7803631d-f181-4500-b3a2-1b68ba2a75fd",
      "address":"10.0.0.5",
      "status_description":null,
      "id":"48a471ea-64f1-4eb6-9be7-dae6bbe40a0f"
   }
}
    `)
	})

	options := UpdateOpts{AdminStateUp: false}

	_, err := Update(fake.ServiceClient(), "332abe93-f488-41ba-870b-2ac66be7f853", options).Extract()
	th.AssertNoErr(t, err)
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/lb/members/332abe93-f488-41ba-870b-2ac66be7f853", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusNoContent)
	})

	res := Delete(fake.ServiceClient(), "332abe93-f488-41ba-870b-2ac66be7f853")
	th.AssertNoErr(t, res.Err)
}
