package testing

import (
	"fmt"
	"net/http"
	"testing"

	fake "github.com/gophercloud/gophercloud/openstack/networking/v2/common"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/vpnaas/ikepolicies"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/vpn/ikepolicies", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
    "ikepolicy":{
        "name": "policy",
        "description": "IKE policy",
		"tenant_id": "9145d91459d248b1b02fdaca97c6a75d",
		"ike_version": "v2"
    }
}
      `)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, `
{
    "ikepolicy":{
        "name": "policy",
        "tenant_id": "9145d91459d248b1b02fdaca97c6a75d",
        "project_id": "9145d91459d248b1b02fdaca97c6a75d",
        "id": "f2b08c1e-aa81-4668-8ae1-1401bcb0576c",
        "description": "IKE policy",
		"auth_algorithm": "sha1",
		"encryption_algorithm": "aes-128",
		"pfs": "Group5",
		"lifetime": {
			"value": 3600,
			"units": "seconds"
		},
		"phase1_negotiation_mode": "main",
		"ike_version": "v2"
    }
}
        `)
	})

	options := ikepolicies.CreateOpts{
		TenantID:    "9145d91459d248b1b02fdaca97c6a75d",
		Name:        "policy",
		Description: "IKE policy",
		IKEVersion:  ikepolicies.IKEVersionv2,
	}

	actual, err := ikepolicies.Create(fake.ServiceClient(), options).Extract()
	th.AssertNoErr(t, err)
	expectedLifetime := ikepolicies.Lifetime{
		Units: "seconds",
		Value: 3600,
	}
	expected := ikepolicies.Policy{
		AuthAlgorithm:         "sha1",
		IKEVersion:            "v2",
		TenantID:              "9145d91459d248b1b02fdaca97c6a75d",
		Phase1NegotiationMode: "main",
		PFS:                   "Group5",
		EncryptionAlgorithm:   "aes-128",
		Description:           "IKE policy",
		Name:                  "policy",
		ID:                    "f2b08c1e-aa81-4668-8ae1-1401bcb0576c",
		Lifetime:              expectedLifetime,
		ProjectID:             "9145d91459d248b1b02fdaca97c6a75d",
	}
	th.AssertDeepEquals(t, expected, *actual)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/vpn/ikepolicies/5c561d9d-eaea-45f6-ae3e-08d1a7080828", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "ikepolicy":{
        "name": "policy",
        "tenant_id": "9145d91459d248b1b02fdaca97c6a75d",
        "project_id": "9145d91459d248b1b02fdaca97c6a75d",
        "id": "5c561d9d-eaea-45f6-ae3e-08d1a7080828",
        "description": "IKE policy",
		"auth_algorithm": "sha1",
		"encryption_algorithm": "aes-128",
		"pfs": "Group5",
		"lifetime": {
			"value": 3600,
			"units": "seconds"
		},
		"phase1_negotiation_mode": "main",
		"ike_version": "v2"
    }
}
        `)
	})

	actual, err := ikepolicies.Get(fake.ServiceClient(), "5c561d9d-eaea-45f6-ae3e-08d1a7080828").Extract()
	th.AssertNoErr(t, err)
	expectedLifetime := ikepolicies.Lifetime{
		Units: "seconds",
		Value: 3600,
	}
	expected := ikepolicies.Policy{
		AuthAlgorithm:         "sha1",
		IKEVersion:            "v2",
		TenantID:              "9145d91459d248b1b02fdaca97c6a75d",
		ProjectID:             "9145d91459d248b1b02fdaca97c6a75d",
		Phase1NegotiationMode: "main",
		PFS:                   "Group5",
		EncryptionAlgorithm:   "aes-128",
		Description:           "IKE policy",
		Name:                  "policy",
		ID:                    "5c561d9d-eaea-45f6-ae3e-08d1a7080828",
		Lifetime:              expectedLifetime,
	}
	th.AssertDeepEquals(t, expected, *actual)
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/vpn/ikepolicies/5c561d9d-eaea-45f6-ae3e-08d1a7080828", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusNoContent)
	})

	res := ikepolicies.Delete(fake.ServiceClient(), "5c561d9d-eaea-45f6-ae3e-08d1a7080828")
	th.AssertNoErr(t, res.Err)
}

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/vpn/ikepolicies", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
		{
	"ikepolicies": [
		{
			"name": "policy",
			"tenant_id": "9145d91459d248b1b02fdaca97c6a75d",
			"project_id": "9145d91459d248b1b02fdaca97c6a75d",
			"id": "5c561d9d-eaea-45f6-ae3e-08d1a7080828",
			"description": "IKE policy",
			"auth_algorithm": "sha1",
			"encryption_algorithm": "aes-128",
			"pfs": "Group5",
			"lifetime": {
				"value": 3600,
				"units": "seconds"
			},
			"phase1_negotiation_mode": "main",
			"ike_version": "v2"
		}
	]
}
	  `)
	})

	count := 0

	ikepolicies.List(fake.ServiceClient(), ikepolicies.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ikepolicies.ExtractPolicies(page)
		if err != nil {
			t.Errorf("Failed to extract members: %v", err)
			return false, err
		}
		expectedLifetime := ikepolicies.Lifetime{
			Units: "seconds",
			Value: 3600,
		}
		expected := []ikepolicies.Policy{
			{
				AuthAlgorithm:         "sha1",
				IKEVersion:            "v2",
				TenantID:              "9145d91459d248b1b02fdaca97c6a75d",
				ProjectID:             "9145d91459d248b1b02fdaca97c6a75d",
				Phase1NegotiationMode: "main",
				PFS:                   "Group5",
				EncryptionAlgorithm:   "aes-128",
				Description:           "IKE policy",
				Name:                  "policy",
				ID:                    "5c561d9d-eaea-45f6-ae3e-08d1a7080828",
				Lifetime:              expectedLifetime,
			},
		}

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})

	if count != 1 {
		t.Errorf("Expected 1 page, got %d", count)
	}
}

func TestUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/vpn/ikepolicies/5c561d9d-eaea-45f6-ae3e-08d1a7080828", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
			{
				"ikepolicy":{
			"name": "updatedname",
			"description": "updated policy",
			"lifetime": {
			"value": 7000
			}
			}
			}
			`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "ikepolicy": {
        "name": "updatedname",
        "transform_protocol": "esp",
        "auth_algorithm": "sha1",
        "encapsulation_mode": "tunnel",
        "encryption_algorithm": "aes-128",
        "pfs": "group5",
        "tenant_id": "b4eedccc6fb74fa8a7ad6b08382b852b",
        "project_id": "b4eedccc6fb74fa8a7ad6b08382b852b",
        "lifetime": {
            "units": "seconds",
            "value": 7000
        },
        "id": "5c561d9d-eaea-45f6-ae3e-08d1a7080828",
        "description": "updated policy"
    }
}
`)
	})

	updatedName := "updatedname"
	updatedDescription := "updated policy"
	options := ikepolicies.UpdateOpts{
		Name:        &updatedName,
		Description: &updatedDescription,
		Lifetime: &ikepolicies.LifetimeUpdateOpts{
			Value: 7000,
		},
	}

	actual, err := ikepolicies.Update(fake.ServiceClient(), "5c561d9d-eaea-45f6-ae3e-08d1a7080828", options).Extract()
	th.AssertNoErr(t, err)
	expectedLifetime := ikepolicies.Lifetime{
		Units: "seconds",
		Value: 7000,
	}
	expected := ikepolicies.Policy{
		TenantID:            "b4eedccc6fb74fa8a7ad6b08382b852b",
		ProjectID:           "b4eedccc6fb74fa8a7ad6b08382b852b",
		Name:                "updatedname",
		AuthAlgorithm:       "sha1",
		EncryptionAlgorithm: "aes-128",
		PFS:                 "group5",
		Description:         "updated policy",
		Lifetime:            expectedLifetime,
		ID:                  "5c561d9d-eaea-45f6-ae3e-08d1a7080828",
	}
	th.AssertDeepEquals(t, expected, *actual)
}
