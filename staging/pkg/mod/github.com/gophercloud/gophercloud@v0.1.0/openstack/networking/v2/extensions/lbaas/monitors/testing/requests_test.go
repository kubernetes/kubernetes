package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud"
	fake "github.com/gophercloud/gophercloud/openstack/networking/v2/common"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas/monitors"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/lb/health_monitors", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
   "health_monitors":[
      {
         "admin_state_up":true,
         "tenant_id":"83657cfcdfe44cd5920adaf26c48ceea",
         "delay":10,
         "max_retries":1,
         "timeout":1,
         "type":"PING",
         "id":"466c8345-28d8-4f84-a246-e04380b0461d"
      },
      {
         "admin_state_up":true,
         "tenant_id":"83657cfcdfe44cd5920adaf26c48ceea",
         "delay":5,
         "expected_codes":"200",
         "max_retries":2,
         "http_method":"GET",
         "timeout":2,
         "url_path":"/",
         "type":"HTTP",
         "id":"5d4b5228-33b0-4e60-b225-9b727c1a20e7"
      }
   ]
}
			`)
	})

	count := 0

	monitors.List(fake.ServiceClient(), monitors.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := monitors.ExtractMonitors(page)
		if err != nil {
			t.Errorf("Failed to extract monitors: %v", err)
			return false, err
		}

		expected := []monitors.Monitor{
			{
				AdminStateUp: true,
				TenantID:     "83657cfcdfe44cd5920adaf26c48ceea",
				Delay:        10,
				MaxRetries:   1,
				Timeout:      1,
				Type:         "PING",
				ID:           "466c8345-28d8-4f84-a246-e04380b0461d",
			},
			{
				AdminStateUp:  true,
				TenantID:      "83657cfcdfe44cd5920adaf26c48ceea",
				Delay:         5,
				ExpectedCodes: "200",
				MaxRetries:    2,
				Timeout:       2,
				URLPath:       "/",
				Type:          "HTTP",
				HTTPMethod:    "GET",
				ID:            "5d4b5228-33b0-4e60-b225-9b727c1a20e7",
			},
		}

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})

	if count != 1 {
		t.Errorf("Expected 1 page, got %d", count)
	}
}

func TestDelayMustBeGreaterOrEqualThanTimeout(t *testing.T) {
	_, err := monitors.Create(fake.ServiceClient(), monitors.CreateOpts{
		Type:          "HTTP",
		Delay:         1,
		Timeout:       10,
		MaxRetries:    5,
		URLPath:       "/check",
		ExpectedCodes: "200-299",
	}).Extract()

	if err == nil {
		t.Fatalf("Expected error, got none")
	}

	_, err = monitors.Update(fake.ServiceClient(), "453105b9-1754-413f-aab1-55f1af620750", monitors.UpdateOpts{
		Delay:   1,
		Timeout: 10,
	}).Extract()

	if err == nil {
		t.Fatalf("Expected error, got none")
	}
}

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/lb/health_monitors", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
   "health_monitor":{
      "type":"HTTP",
      "tenant_id":"453105b9-1754-413f-aab1-55f1af620750",
      "delay":20,
      "timeout":10,
      "max_retries":5,
      "url_path":"/check",
      "expected_codes":"200-299"
   }
}
			`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, `
{
   "health_monitor":{
      "id":"f3eeab00-8367-4524-b662-55e64d4cacb5",
      "tenant_id":"453105b9-1754-413f-aab1-55f1af620750",
      "type":"HTTP",
      "delay":20,
      "timeout":10,
      "max_retries":5,
      "http_method":"GET",
      "url_path":"/check",
      "expected_codes":"200-299",
      "admin_state_up":true,
      "status":"ACTIVE"
   }
}
		`)
	})

	_, err := monitors.Create(fake.ServiceClient(), monitors.CreateOpts{
		Type:          "HTTP",
		TenantID:      "453105b9-1754-413f-aab1-55f1af620750",
		Delay:         20,
		Timeout:       10,
		MaxRetries:    5,
		URLPath:       "/check",
		ExpectedCodes: "200-299",
	}).Extract()

	th.AssertNoErr(t, err)
}

func TestRequiredCreateOpts(t *testing.T) {
	res := monitors.Create(fake.ServiceClient(), monitors.CreateOpts{})
	if res.Err == nil {
		t.Fatalf("Expected error, got none")
	}
	res = monitors.Create(fake.ServiceClient(), monitors.CreateOpts{Type: monitors.TypeHTTP})
	if res.Err == nil {
		t.Fatalf("Expected error, got none")
	}
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/lb/health_monitors/f3eeab00-8367-4524-b662-55e64d4cacb5", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
   "health_monitor":{
      "id":"f3eeab00-8367-4524-b662-55e64d4cacb5",
      "tenant_id":"453105b9-1754-413f-aab1-55f1af620750",
      "type":"HTTP",
      "delay":20,
      "timeout":10,
      "max_retries":5,
      "http_method":"GET",
      "url_path":"/check",
      "expected_codes":"200-299",
      "admin_state_up":true,
      "status":"ACTIVE"
   }
}
			`)
	})

	hm, err := monitors.Get(fake.ServiceClient(), "f3eeab00-8367-4524-b662-55e64d4cacb5").Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, "f3eeab00-8367-4524-b662-55e64d4cacb5", hm.ID)
	th.AssertEquals(t, "453105b9-1754-413f-aab1-55f1af620750", hm.TenantID)
	th.AssertEquals(t, "HTTP", hm.Type)
	th.AssertEquals(t, 20, hm.Delay)
	th.AssertEquals(t, 10, hm.Timeout)
	th.AssertEquals(t, 5, hm.MaxRetries)
	th.AssertEquals(t, "GET", hm.HTTPMethod)
	th.AssertEquals(t, "/check", hm.URLPath)
	th.AssertEquals(t, "200-299", hm.ExpectedCodes)
	th.AssertEquals(t, true, hm.AdminStateUp)
	th.AssertEquals(t, "ACTIVE", hm.Status)
}

func TestUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/lb/health_monitors/b05e44b5-81f9-4551-b474-711a722698f7", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
   "health_monitor":{
      "delay": 30,
      "timeout": 20,
      "max_retries": 10,
      "url_path": "/another_check",
      "expected_codes": "301",
			"admin_state_up": true
   }
}
			`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusAccepted)

		fmt.Fprintf(w, `
{
    "health_monitor": {
        "admin_state_up": true,
        "tenant_id": "4fd44f30292945e481c7b8a0c8908869",
        "delay": 30,
        "max_retries": 10,
        "http_method": "GET",
        "timeout": 20,
        "pools": [
            {
                "status": "PENDING_CREATE",
                "status_description": null,
                "pool_id": "6e55751f-6ad4-4e53-b8d4-02e442cd21df"
            }
        ],
        "type": "PING",
        "id": "b05e44b5-81f9-4551-b474-711a722698f7"
    }
}
		`)
	})

	_, err := monitors.Update(fake.ServiceClient(), "b05e44b5-81f9-4551-b474-711a722698f7", monitors.UpdateOpts{
		Delay:         30,
		Timeout:       20,
		MaxRetries:    10,
		URLPath:       "/another_check",
		ExpectedCodes: "301",
		AdminStateUp:  gophercloud.Enabled,
	}).Extract()

	th.AssertNoErr(t, err)
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/lb/health_monitors/b05e44b5-81f9-4551-b474-711a722698f7", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusNoContent)
	})

	res := monitors.Delete(fake.ServiceClient(), "b05e44b5-81f9-4551-b474-711a722698f7")
	th.AssertNoErr(t, res.Err)
}
