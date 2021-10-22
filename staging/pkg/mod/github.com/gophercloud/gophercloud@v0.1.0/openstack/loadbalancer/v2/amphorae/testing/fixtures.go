package testing

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud/openstack/loadbalancer/v2/amphorae"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

// AmphoraeListBody contains the canned body of a amphora list response.
const AmphoraeListBody = `
{
    "amphorae": [
        {
            "cached_zone": "nova",
            "cert_busy": false,
            "cert_expiration": "2020-08-08T23:44:31",
            "compute_id": "667bb225-69aa-44b1-8908-694dc624c267",
            "created_at": "2018-08-09T23:44:31",
            "ha_ip": "10.0.0.6",
            "ha_port_id": "35254b63-9361-4561-9b8f-2bb4e3be60e3",
            "id": "45f40289-0551-483a-b089-47214bc2a8a4",
            "image_id": "5d1aed06-2624-43f5-a413-9212263c3d53",
            "lb_network_ip": "192.168.0.6",
            "loadbalancer_id": "882f2a9d-9d53-4bd0-b0e9-08e9d0de11f9",
            "role": "MASTER",
            "status": "READY",
            "updated_at": "2018-08-09T23:51:06",
            "vrrp_id": 1,
            "vrrp_interface": "eth1",
            "vrrp_ip": "10.0.0.4",
            "vrrp_port_id": "dcf0c8b5-6a08-4658-997d-eac97f2b9bbd",
            "vrrp_priority": 100
        },
        {
            "cached_zone": "nova",
            "cert_busy": false,
            "cert_expiration": "2020-08-08T23:44:30",
            "compute_id": "9cd0f9a2-fe12-42fc-a7e3-5b6fbbe20395",
            "created_at": "2018-08-09T23:44:31",
            "ha_ip": "10.0.0.6",
            "ha_port_id": "35254b63-9361-4561-9b8f-2bb4e3be60e3",
            "id": "7f890893-ced0-46ed-8697-33415d070e5a",
            "image_id": "5d1aed06-2624-43f5-a413-9212263c3d53",
            "lb_network_ip": "192.168.0.17",
            "loadbalancer_id": "882f2a9d-9d53-4bd0-b0e9-08e9d0de11f9",
            "role": "BACKUP",
            "status": "READY",
            "updated_at": "2018-08-09T23:51:06",
            "vrrp_id": 1,
            "vrrp_interface": "eth1",
            "vrrp_ip": "10.0.0.21",
            "vrrp_port_id": "13c88c77-207d-4f85-8f7a-84344592e367",
            "vrrp_priority": 90
        }
    ],
    "amphorae_links": []
}
`

const SingleAmphoraBody = `
{
	"amphora": {
		"cached_zone": "nova",
		"cert_busy": false,
		"cert_expiration": "2020-08-08T23:44:31",
		"compute_id": "667bb225-69aa-44b1-8908-694dc624c267",
		"created_at": "2018-08-09T23:44:31",
		"ha_ip": "10.0.0.6",
		"ha_port_id": "35254b63-9361-4561-9b8f-2bb4e3be60e3",
		"id": "45f40289-0551-483a-b089-47214bc2a8a4",
		"image_id": "5d1aed06-2624-43f5-a413-9212263c3d53",
		"lb_network_ip": "192.168.0.6",
		"loadbalancer_id": "882f2a9d-9d53-4bd0-b0e9-08e9d0de11f9",
		"role": "MASTER",
		"status": "READY",
		"updated_at": "2018-08-09T23:51:06",
		"vrrp_id": 1,
		"vrrp_interface": "eth1",
		"vrrp_ip": "10.0.0.4",
		"vrrp_port_id": "dcf0c8b5-6a08-4658-997d-eac97f2b9bbd",
		"vrrp_priority": 100
	}
}
`

// FirstAmphora is the first resource in the List request.
var FirstAmphora = amphorae.Amphora{
	CachedZone:     "nova",
	CertBusy:       false,
	CertExpiration: time.Date(2020, 8, 8, 23, 44, 31, 0, time.UTC),
	ComputeID:      "667bb225-69aa-44b1-8908-694dc624c267",
	CreatedAt:      time.Date(2018, 8, 9, 23, 44, 31, 0, time.UTC),
	HAIP:           "10.0.0.6",
	HAPortID:       "35254b63-9361-4561-9b8f-2bb4e3be60e3",
	ID:             "45f40289-0551-483a-b089-47214bc2a8a4",
	ImageID:        "5d1aed06-2624-43f5-a413-9212263c3d53",
	LBNetworkIP:    "192.168.0.6",
	LoadbalancerID: "882f2a9d-9d53-4bd0-b0e9-08e9d0de11f9",
	Role:           "MASTER",
	Status:         "READY",
	UpdatedAt:      time.Date(2018, 8, 9, 23, 51, 6, 0, time.UTC),
	VRRPID:         1,
	VRRPInterface:  "eth1",
	VRRPIP:         "10.0.0.4",
	VRRPPortID:     "dcf0c8b5-6a08-4658-997d-eac97f2b9bbd",
	VRRPPriority:   100,
}

// SecondAmphora is the second resource in the List request.
var SecondAmphora = amphorae.Amphora{
	CachedZone:     "nova",
	CertBusy:       false,
	CertExpiration: time.Date(2020, 8, 8, 23, 44, 30, 0, time.UTC),
	ComputeID:      "9cd0f9a2-fe12-42fc-a7e3-5b6fbbe20395",
	CreatedAt:      time.Date(2018, 8, 9, 23, 44, 31, 0, time.UTC),
	HAIP:           "10.0.0.6",
	HAPortID:       "35254b63-9361-4561-9b8f-2bb4e3be60e3",
	ID:             "7f890893-ced0-46ed-8697-33415d070e5a",
	ImageID:        "5d1aed06-2624-43f5-a413-9212263c3d53",
	LBNetworkIP:    "192.168.0.17",
	LoadbalancerID: "882f2a9d-9d53-4bd0-b0e9-08e9d0de11f9",
	Role:           "BACKUP",
	Status:         "READY",
	UpdatedAt:      time.Date(2018, 8, 9, 23, 51, 6, 0, time.UTC),
	VRRPID:         1,
	VRRPInterface:  "eth1",
	VRRPIP:         "10.0.0.21",
	VRRPPortID:     "13c88c77-207d-4f85-8f7a-84344592e367",
	VRRPPriority:   90,
}

// ExpectedAmphoraeSlice is the slice of amphorae expected to be returned from ListResponse.
var ExpectedAmphoraeSlice = []amphorae.Amphora{FirstAmphora, SecondAmphora}

// HandleAmphoraListSuccessfully sets up the test server to respond to a amphorae List request.
func HandleAmphoraListSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/octavia/amphorae", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		r.ParseForm()
		marker := r.Form.Get("marker")
		switch marker {
		case "":
			fmt.Fprintf(w, AmphoraeListBody)
		case "7f890893-ced0-46ed-8697-33415d070e5a":
			fmt.Fprintf(w, `{ "amphorae": [] }`)
		default:
			t.Fatalf("/v2.0/octavia/amphorae invoked with unexpected marker=[%s]", marker)
		}
	})
}

// HandleAmphoraGetSuccessfully sets up the test server to respond to am amphora Get request.
func HandleAmphoraGetSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/octavia/amphorae/45f40289-0551-483a-b089-47214bc2a8a4", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		fmt.Fprintf(w, SingleAmphoraBody)
	})
}
