package testing

import (
	"encoding/json"
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/clustering/v1/webhooks"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

func TestWebhookTrigger(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
			{
				"action": "290c44fa-c60f-4d75-a0eb-87433ba982a3"
			}`)
	})

	triggerOpts := webhooks.TriggerOpts{
		V: "1",
		Params: map[string]string{
			"foo": "bar",
			"bar": "baz",
		},
	}
	result, err := webhooks.Trigger(fake.ServiceClient(), "f93f83f6-762b-41b6-b757-80507834d394", triggerOpts).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, result, "290c44fa-c60f-4d75-a0eb-87433ba982a3")
}

// Test webhook with params that generates query strings
func TestWebhookParams(t *testing.T) {
	triggerOpts := webhooks.TriggerOpts{
		V: "1",
		Params: map[string]string{
			"foo": "bar",
			"bar": "baz",
		},
	}
	expected := "?V=1&bar=baz&foo=bar"
	actual, err := triggerOpts.ToWebhookTriggerQuery()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, actual, expected)
}

// Nagative test case for returning invalid type (integer) for action id
func TestWebhooksInvalidAction(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
			{
				"action": 123
			}`)
	})

	triggerOpts := webhooks.TriggerOpts{
		V: "1",
		Params: map[string]string{
			"foo": "bar",
			"bar": "baz",
		},
	}
	_, err := webhooks.Trigger(fake.ServiceClient(), "f93f83f6-762b-41b6-b757-80507834d394", triggerOpts).Extract()
	isValid := err.(*json.UnmarshalTypeError) == nil
	th.AssertEquals(t, false, isValid)
}

// Negative test case for passing empty TriggerOpt
func TestWebhookTriggerInvalidEmptyOpt(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
			{
				"action": "290c44fa-c60f-4d75-a0eb-87433ba982a3"
			}`)
	})

	_, err := webhooks.Trigger(fake.ServiceClient(), "f93f83f6-762b-41b6-b757-80507834d394", webhooks.TriggerOpts{}).Extract()
	if err == nil {
		t.Errorf("Expected error without V param")
	}
}

// Negative test case for passing in nil for TriggerOpt
func TestWebhookTriggerInvalidNilOpt(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
			{
				"action": "290c44fa-c60f-4d75-a0eb-87433ba982a3"
			}`)
	})

	_, err := webhooks.Trigger(fake.ServiceClient(), "f93f83f6-762b-41b6-b757-80507834d394", nil).Extract()

	if err == nil {
		t.Errorf("Expected error with nil param")
	}
}
