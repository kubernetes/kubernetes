package testing

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/services"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

// ServiceListBodyPre253 represents a raw service list from the Compute API
// with microversion older than 2.53.
const ServiceListBodyPre253 = `
{
    "services": [
        {
            "id": 1,
            "binary": "nova-scheduler",
            "disabled_reason": "test1",
            "host": "host1",
            "state": "up",
            "status": "disabled",
            "updated_at": "2012-10-29T13:42:02.000000",
            "forced_down": false,
            "zone": "internal"
        },
        {
            "id": 2,
            "binary": "nova-compute",
            "disabled_reason": "test2",
            "host": "host1",
            "state": "up",
            "status": "disabled",
            "updated_at": "2012-10-29T13:42:05.000000",
            "forced_down": false,
            "zone": "nova"
        },
        {
            "id": 3,
            "binary": "nova-scheduler",
            "disabled_reason": null,
            "host": "host2",
            "state": "down",
            "status": "enabled",
            "updated_at": "2012-09-19T06:55:34.000000",
            "forced_down": false,
            "zone": "internal"
        },
        {
            "id": 4,
            "binary": "nova-compute",
            "disabled_reason": "test4",
            "host": "host2",
            "state": "down",
            "status": "disabled",
            "updated_at": "2012-09-18T08:03:38.000000",
            "forced_down": false,
            "zone": "nova"
        }
    ]
}
`

var (
	// FirstFakeServicePre253 represents the first service from the
	// ServiceListBodyPre253.
	FirstFakeServicePre253 = services.Service{
		Binary:         "nova-scheduler",
		DisabledReason: "test1",
		Host:           "host1",
		ID:             "1",
		State:          "up",
		Status:         "disabled",
		UpdatedAt:      time.Date(2012, 10, 29, 13, 42, 2, 0, time.UTC),
		Zone:           "internal",
	}

	// SecondFakeServicePre253 represents the second service from the
	// ServiceListBodyPre253.
	SecondFakeServicePre253 = services.Service{
		Binary:         "nova-compute",
		DisabledReason: "test2",
		Host:           "host1",
		ID:             "2",
		State:          "up",
		Status:         "disabled",
		UpdatedAt:      time.Date(2012, 10, 29, 13, 42, 5, 0, time.UTC),
		Zone:           "nova",
	}

	// ThirdFakeServicePre253 represents the third service from the
	// ServiceListBodyPre253.
	ThirdFakeServicePre253 = services.Service{
		Binary:         "nova-scheduler",
		DisabledReason: "",
		Host:           "host2",
		ID:             "3",
		State:          "down",
		Status:         "enabled",
		UpdatedAt:      time.Date(2012, 9, 19, 6, 55, 34, 0, time.UTC),
		Zone:           "internal",
	}

	// FourthFakeServicePre253 represents the fourth service from the
	// ServiceListBodyPre253.
	FourthFakeServicePre253 = services.Service{
		Binary:         "nova-compute",
		DisabledReason: "test4",
		Host:           "host2",
		ID:             "4",
		State:          "down",
		Status:         "disabled",
		UpdatedAt:      time.Date(2012, 9, 18, 8, 3, 38, 0, time.UTC),
		Zone:           "nova",
	}
)

// ServiceListBody represents a raw service list result with Pike+ release.
const ServiceListBody = `
{
    "services": [
        {
            "id": "4c720fa0-02c3-4834-8279-9eecf9edb6cb",
            "binary": "nova-scheduler",
            "disabled_reason": "test1",
            "host": "host1",
            "state": "up",
            "status": "disabled",
            "updated_at": "2012-10-29T13:42:02.000000",
            "forced_down": false,
            "zone": "internal"
        },
        {
            "id": "1fdfec3e-ee03-4e36-b99b-71cf2967b70c",
            "binary": "nova-compute",
            "disabled_reason": "test2",
            "host": "host1",
            "state": "up",
            "status": "disabled",
            "updated_at": "2012-10-29T13:42:05.000000",
            "forced_down": false,
            "zone": "nova"
        },
        {
            "id": "bd0b2e30-809e-4160-bd3d-f23ca30e9b68",
            "binary": "nova-scheduler",
            "disabled_reason": null,
            "host": "host2",
            "state": "down",
            "status": "enabled",
            "updated_at": "2012-09-19T06:55:34.000000",
            "forced_down": false,
            "zone": "internal"
        },
        {
            "id": "fe41c476-33e2-4ac3-ad21-3ffaf1b9c644",
            "binary": "nova-compute",
            "disabled_reason": "test4",
            "host": "host2",
            "state": "down",
            "status": "disabled",
            "updated_at": "2012-09-18T08:03:38.000000",
            "forced_down": false,
            "zone": "nova"
        }
    ]
}
`

var (
	// FirstFakeService represents the first service from the ServiceListBody.
	FirstFakeService = services.Service{
		Binary:         "nova-scheduler",
		DisabledReason: "test1",
		Host:           "host1",
		ID:             "4c720fa0-02c3-4834-8279-9eecf9edb6cb",
		State:          "up",
		Status:         "disabled",
		UpdatedAt:      time.Date(2012, 10, 29, 13, 42, 2, 0, time.UTC),
		Zone:           "internal",
	}

	// SecondFakeService represents the second service from the ServiceListBody.
	SecondFakeService = services.Service{
		Binary:         "nova-compute",
		DisabledReason: "test2",
		Host:           "host1",
		ID:             "1fdfec3e-ee03-4e36-b99b-71cf2967b70c",
		State:          "up",
		Status:         "disabled",
		UpdatedAt:      time.Date(2012, 10, 29, 13, 42, 5, 0, time.UTC),
		Zone:           "nova",
	}

	// ThirdFakeService represents the third service from the ServiceListBody.
	ThirdFakeService = services.Service{
		Binary:         "nova-scheduler",
		DisabledReason: "",
		Host:           "host2",
		ID:             "bd0b2e30-809e-4160-bd3d-f23ca30e9b68",
		State:          "down",
		Status:         "enabled",
		UpdatedAt:      time.Date(2012, 9, 19, 6, 55, 34, 0, time.UTC),
		Zone:           "internal",
	}

	// FourthFakeService represents the fourth service from the ServiceListBody.
	FourthFakeService = services.Service{
		Binary:         "nova-compute",
		DisabledReason: "test4",
		Host:           "host2",
		ID:             "fe41c476-33e2-4ac3-ad21-3ffaf1b9c644",
		State:          "down",
		Status:         "disabled",
		UpdatedAt:      time.Date(2012, 9, 18, 8, 3, 38, 0, time.UTC),
		Zone:           "nova",
	}
)

// HandleListPre253Successfully configures the test server to respond to a List
// request to a Compute server API pre 2.53 microversion release.
func HandleListPre253Successfully(t *testing.T) {
	th.Mux.HandleFunc("/os-services", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, ServiceListBodyPre253)
	})
}

// HandleListSuccessfully configures the test server to respond to a List
// request to a Compute server with Pike+ release.
func HandleListSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/os-services", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, ServiceListBody)
	})
}
