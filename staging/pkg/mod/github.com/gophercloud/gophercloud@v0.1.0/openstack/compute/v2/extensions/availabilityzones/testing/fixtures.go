package testing

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	az "github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/availabilityzones"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

const GetOutput = `
{
    "availabilityZoneInfo": [
        {
            "hosts": null,
            "zoneName": "nova",
            "zoneState": {
                "available": true
            }
        }
    ]
}
`

const GetDetailOutput = `
{
    "availabilityZoneInfo": [
        {
            "hosts": {
                "localhost": {
                    "nova-cert": {
                        "active": true,
                        "available": false,
                        "updated_at": "2017-10-14T17:03:39.000000"
                    },
                    "nova-conductor": {
                        "active": true,
                        "available": false,
                        "updated_at": "2017-10-14T17:04:09.000000"
                    },
                    "nova-consoleauth": {
                        "active": true,
                        "available": false,
                        "updated_at": "2017-10-14T17:04:18.000000"
                    },
                    "nova-scheduler": {
                        "active": true,
                        "available": false,
                        "updated_at": "2017-10-14T17:04:30.000000"
                    }
                },
                "openstack-acc-tests.novalocal": {
                    "nova-cert": {
                        "active": true,
                        "available": true,
                        "updated_at": "2018-01-04T04:11:19.000000"
                    },
                    "nova-conductor": {
                        "active": true,
                        "available": true,
                        "updated_at": "2018-01-04T04:11:22.000000"
                    },
                    "nova-consoleauth": {
                        "active": true,
                        "available": true,
                        "updated_at": "2018-01-04T04:11:20.000000"
                    },
                    "nova-scheduler": {
                        "active": true,
                        "available": true,
                        "updated_at": "2018-01-04T04:11:23.000000"
                    }
                }
            },
            "zoneName": "internal",
            "zoneState": {
                "available": true
            }
        },
        {
            "hosts": {
                "openstack-acc-tests.novalocal": {
                    "nova-compute": {
                        "active": true,
                        "available": true,
                        "updated_at": "2018-01-04T04:11:23.000000"
                    }
                }
            },
            "zoneName": "nova",
            "zoneState": {
                "available": true
            }
        }
    ]
}`

var AZResult = []az.AvailabilityZone{
	{
		Hosts:     nil,
		ZoneName:  "nova",
		ZoneState: az.ZoneState{Available: true},
	},
}

var AZDetailResult = []az.AvailabilityZone{
	{
		Hosts: az.Hosts{
			"localhost": az.Services{
				"nova-cert": az.ServiceState{
					Active:    true,
					Available: false,
					UpdatedAt: time.Date(2017, 10, 14, 17, 3, 39, 0, time.UTC),
				},
				"nova-conductor": az.ServiceState{
					Active:    true,
					Available: false,
					UpdatedAt: time.Date(2017, 10, 14, 17, 4, 9, 0, time.UTC),
				},
				"nova-consoleauth": az.ServiceState{
					Active:    true,
					Available: false,
					UpdatedAt: time.Date(2017, 10, 14, 17, 4, 18, 0, time.UTC),
				},
				"nova-scheduler": az.ServiceState{
					Active:    true,
					Available: false,
					UpdatedAt: time.Date(2017, 10, 14, 17, 4, 30, 0, time.UTC),
				},
			},
			"openstack-acc-tests.novalocal": az.Services{
				"nova-cert": az.ServiceState{
					Active:    true,
					Available: true,
					UpdatedAt: time.Date(2018, 1, 4, 4, 11, 19, 0, time.UTC),
				},
				"nova-conductor": az.ServiceState{
					Active:    true,
					Available: true,
					UpdatedAt: time.Date(2018, 1, 4, 4, 11, 22, 0, time.UTC),
				},
				"nova-consoleauth": az.ServiceState{
					Active:    true,
					Available: true,
					UpdatedAt: time.Date(2018, 1, 4, 4, 11, 20, 0, time.UTC),
				},
				"nova-scheduler": az.ServiceState{
					Active:    true,
					Available: true,
					UpdatedAt: time.Date(2018, 1, 4, 4, 11, 23, 0, time.UTC),
				},
			},
		},
		ZoneName:  "internal",
		ZoneState: az.ZoneState{Available: true},
	},
	{
		Hosts: az.Hosts{
			"openstack-acc-tests.novalocal": az.Services{
				"nova-compute": az.ServiceState{
					Active:    true,
					Available: true,
					UpdatedAt: time.Date(2018, 1, 4, 4, 11, 23, 0, time.UTC),
				},
			},
		},
		ZoneName:  "nova",
		ZoneState: az.ZoneState{Available: true},
	},
}

// HandleGetSuccessfully configures the test server to respond to a Get request
// for availability zone information.
func HandleGetSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/os-availability-zone", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, GetOutput)
	})
}

// HandleGetDetailSuccessfully configures the test server to respond to a Get request
// for detailed availability zone information.
func HandleGetDetailSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/os-availability-zone/detail", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, GetDetailOutput)
	})
}
