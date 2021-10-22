package testing

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud/openstack/blockstorage/extensions/services"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

// ServiceListBody is sample response to the List call
const ServiceListBody = `
{
    "services": [{
        "status": "enabled",
        "binary": "cinder-scheduler",
        "zone": "nova",
        "state": "up",
        "updated_at": "2017-06-29T05:50:35.000000",
        "host": "devstack",
        "disabled_reason": null
    },
    {
        "status": "enabled",
        "binary": "cinder-backup",
        "zone": "nova",
        "state": "up",
        "updated_at": "2017-06-29T05:50:42.000000",
        "host": "devstack",
        "disabled_reason": null
    },
    {
        "status": "enabled",
        "binary": "cinder-volume",
        "zone": "nova",
        "frozen": false,
        "state": "up",
        "updated_at": "2017-06-29T05:50:39.000000",
        "cluster": null,
        "host": "devstack@lvmdriver-1",
        "replication_status": "disabled",
        "active_backend_id": null,
        "disabled_reason": null
    }]
}
`

// First service from the ServiceListBody
var FirstFakeService = services.Service{
	Binary:         "cinder-scheduler",
	DisabledReason: "",
	Host:           "devstack",
	State:          "up",
	Status:         "enabled",
	UpdatedAt:      time.Date(2017, 6, 29, 5, 50, 35, 0, time.UTC),
	Zone:           "nova",
}

// Second service from the ServiceListBody
var SecondFakeService = services.Service{
	Binary:         "cinder-backup",
	DisabledReason: "",
	Host:           "devstack",
	State:          "up",
	Status:         "enabled",
	UpdatedAt:      time.Date(2017, 6, 29, 5, 50, 42, 0, time.UTC),
	Zone:           "nova",
}

// Third service from the ServiceListBody
var ThirdFakeService = services.Service{
	ActiveBackendID:   "",
	Binary:            "cinder-volume",
	Cluster:           "",
	DisabledReason:    "",
	Frozen:            false,
	Host:              "devstack@lvmdriver-1",
	ReplicationStatus: "disabled",
	State:             "up",
	Status:            "enabled",
	UpdatedAt:         time.Date(2017, 6, 29, 5, 50, 39, 0, time.UTC),
	Zone:              "nova",
}

// HandleListSuccessfully configures the test server to respond to a List request.
func HandleListSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/os-services", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, ServiceListBody)
	})
}
