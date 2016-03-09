package policies

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

	th.AssertEquals(t, th.Endpoint()+"v2.0/fw/firewall_policies", rootURL(fake.ServiceClient()))
}

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/fw/firewall_policies", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "firewall_policies": [
        {
            "name": "policy1",
            "firewall_rules": [
                "75452b36-268e-4e75-aaf4-f0e7ed50bc97",
                "c9e77ca0-1bc8-497d-904d-948107873dc6"
            ],
            "tenant_id": "9145d91459d248b1b02fdaca97c6a75d",
            "audited": true,
			"shared": false,
            "id": "f2b08c1e-aa81-4668-8ae1-1401bcb0576c",
            "description": "Firewall policy 1"
        },
        {
            "name": "policy2",
            "firewall_rules": [
                "03d2a6ad-633f-431a-8463-4370d06a22c8"
            ],
            "tenant_id": "9145d91459d248b1b02fdaca97c6a75d",
            "audited": false,
			"shared": true,
            "id": "c854fab5-bdaf-4a86-9359-78de93e5df01",
            "description": "Firewall policy 2"
        }
    ]
}
        `)
	})

	count := 0

	List(fake.ServiceClient(), ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractPolicies(page)
		if err != nil {
			t.Errorf("Failed to extract members: %v", err)
			return false, err
		}

		expected := []Policy{
			Policy{
				Name: "policy1",
				Rules: []string{
					"75452b36-268e-4e75-aaf4-f0e7ed50bc97",
					"c9e77ca0-1bc8-497d-904d-948107873dc6",
				},
				TenantID:    "9145d91459d248b1b02fdaca97c6a75d",
				Audited:     true,
				Shared:      false,
				ID:          "f2b08c1e-aa81-4668-8ae1-1401bcb0576c",
				Description: "Firewall policy 1",
			},
			Policy{
				Name: "policy2",
				Rules: []string{
					"03d2a6ad-633f-431a-8463-4370d06a22c8",
				},
				TenantID:    "9145d91459d248b1b02fdaca97c6a75d",
				Audited:     false,
				Shared:      true,
				ID:          "c854fab5-bdaf-4a86-9359-78de93e5df01",
				Description: "Firewall policy 2",
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

	th.Mux.HandleFunc("/v2.0/fw/firewall_policies", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
    "firewall_policy":{
        "name": "policy",
        "firewall_rules": [
            "98a58c87-76be-ae7c-a74e-b77fffb88d95",
            "11a58c87-76be-ae7c-a74e-b77fffb88a32"
        ],
        "description": "Firewall policy",
		"tenant_id": "9145d91459d248b1b02fdaca97c6a75d",
		"audited": true,
		"shared": false
    }
}
      `)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, `
{
    "firewall_policy":{
        "name": "policy",
        "firewall_rules": [
            "98a58c87-76be-ae7c-a74e-b77fffb88d95",
            "11a58c87-76be-ae7c-a74e-b77fffb88a32"
        ],
        "tenant_id": "9145d91459d248b1b02fdaca97c6a75d",
        "audited": false,
        "id": "f2b08c1e-aa81-4668-8ae1-1401bcb0576c",
        "description": "Firewall policy"
    }
}
        `)
	})

	options := CreateOpts{
		TenantID:    "9145d91459d248b1b02fdaca97c6a75d",
		Name:        "policy",
		Description: "Firewall policy",
		Shared:      No,
		Audited:     Yes,
		Rules: []string{
			"98a58c87-76be-ae7c-a74e-b77fffb88d95",
			"11a58c87-76be-ae7c-a74e-b77fffb88a32",
		},
	}

	_, err := Create(fake.ServiceClient(), options).Extract()
	th.AssertNoErr(t, err)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/fw/firewall_policies/bcab5315-64f6-4ea3-8e58-981cc37c6f61", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "firewall_policy":{
        "name": "www",
        "firewall_rules": [
            "75452b36-268e-4e75-aaf4-f0e7ed50bc97",
            "c9e77ca0-1bc8-497d-904d-948107873dc6",
            "03d2a6ad-633f-431a-8463-4370d06a22c8"
        ],
        "tenant_id": "9145d91459d248b1b02fdaca97c6a75d",
        "audited": false,
        "id": "f2b08c1e-aa81-4668-8ae1-1401bcb0576c",
        "description": "Firewall policy web"
    }
}
        `)
	})

	policy, err := Get(fake.ServiceClient(), "bcab5315-64f6-4ea3-8e58-981cc37c6f61").Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, "www", policy.Name)
	th.AssertEquals(t, "f2b08c1e-aa81-4668-8ae1-1401bcb0576c", policy.ID)
	th.AssertEquals(t, "Firewall policy web", policy.Description)
	th.AssertEquals(t, 3, len(policy.Rules))
	th.AssertEquals(t, "75452b36-268e-4e75-aaf4-f0e7ed50bc97", policy.Rules[0])
	th.AssertEquals(t, "c9e77ca0-1bc8-497d-904d-948107873dc6", policy.Rules[1])
	th.AssertEquals(t, "03d2a6ad-633f-431a-8463-4370d06a22c8", policy.Rules[2])
	th.AssertEquals(t, "9145d91459d248b1b02fdaca97c6a75d", policy.TenantID)
}

func TestUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/fw/firewall_policies/f2b08c1e-aa81-4668-8ae1-1401bcb0576c", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
    "firewall_policy":{
        "name": "policy",
        "firewall_rules": [
            "98a58c87-76be-ae7c-a74e-b77fffb88d95",
            "11a58c87-76be-ae7c-a74e-b77fffb88a32"
        ],
        "description": "Firewall policy"
    }
}
      `)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "firewall_policy":{
        "name": "policy",
        "firewall_rules": [
            "75452b36-268e-4e75-aaf4-f0e7ed50bc97",
            "c9e77ca0-1bc8-497d-904d-948107873dc6",
            "03d2a6ad-633f-431a-8463-4370d06a22c8"
        ],
        "tenant_id": "9145d91459d248b1b02fdaca97c6a75d",
        "audited": false,
        "id": "f2b08c1e-aa81-4668-8ae1-1401bcb0576c",
        "description": "Firewall policy"
    }
}
    `)
	})

	options := UpdateOpts{
		Name:        "policy",
		Description: "Firewall policy",
		Rules: []string{
			"98a58c87-76be-ae7c-a74e-b77fffb88d95",
			"11a58c87-76be-ae7c-a74e-b77fffb88a32",
		},
	}

	_, err := Update(fake.ServiceClient(), "f2b08c1e-aa81-4668-8ae1-1401bcb0576c", options).Extract()
	th.AssertNoErr(t, err)
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/fw/firewall_policies/4ec89077-d057-4a2b-911f-60a3b47ee304", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusNoContent)
	})

	res := Delete(fake.ServiceClient(), "4ec89077-d057-4a2b-911f-60a3b47ee304")
	th.AssertNoErr(t, res.Err)
}
