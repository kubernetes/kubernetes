package testing

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/baremetalintrospection/v1/introspection"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

// IntrospectionListBody contains the canned body of a introspection.IntrospectionList response.
const IntrospectionListBody = `
{
  "introspection": [
    {
      "error": null,
      "finished": true,
      "finished_at": "2017-08-17T11:36:16",
      "links": [
        {
          "href": "http://127.0.0.1:5050/v1/introspection/05ccda19-581b-49bf-8f5a-6ded99701d87",
          "rel": "self"
        }
      ],
      "started_at": "2017-08-17T11:33:43",
      "state": "finished",
      "uuid": "05ccda19-581b-49bf-8f5a-6ded99701d87"
    },
    {
      "error": null,
      "finished": true,
      "finished_at": "2017-08-16T12:24:30",
      "links": [
        {
          "href": "http://127.0.0.1:5050/v1/introspection/c244557e-899f-46fa-a1ff-5b2c6718616b",
          "rel": "self"
        }
      ],
      "started_at": "2017-08-16T12:22:01",
      "state": "finished",
      "uuid": "c244557e-899f-46fa-a1ff-5b2c6718616b"
    }
  ]
}
`

// IntrospectionStatus contains the respnse of a single introspection satus.
const IntrospectionStatus = `
{
  "error": null,
  "finished": true,
  "finished_at": "2017-08-16T12:24:30",
  "links": [
    {
      "href": "http://127.0.0.1:5050/v1/introspection/c244557e-899f-46fa-a1ff-5b2c6718616b",
      "rel": "self"
    }
  ],
  "started_at": "2017-08-16T12:22:01",
  "state": "finished",
  "uuid": "c244557e-899f-46fa-a1ff-5b2c6718616b"
}
`

// IntrospectionDataJSONSample contains sample data reported by the introspection process.
const IntrospectionDataJSONSample = `
{
   "cpu_arch":"x86_64",
   "macs":[
      "52:54:00:4e:3d:30"
   ],
   "root_disk":{
      "rotational":true,
      "vendor":"0x1af4",
      "name":"/dev/vda",
      "hctl":null,
      "wwn_vendor_extension":null,
      "wwn_with_extension":null,
      "model":"",
      "wwn":null,
      "serial":null,
      "size":13958643712
   },
   "interfaces": {
      "eth0": {
         "ip":"172.24.42.100",
         "mac":"52:54:00:4e:3d:30",
         "pxe":true,
         "client_id":null
     }
   },
   "cpus":2,
   "boot_interface":"52:54:00:4e:3d:30",
   "memory_mb":2048,
   "ipmi_address":"192.167.2.134",
   "inventory":{
      "bmc_address":"192.167.2.134",
      "interfaces":[
         {
            "lldp":[],
            "product":"0x0001",
            "vendor":"0x1af4",
            "name":"eth1",
            "has_carrier":true,
            "ipv4_address":"172.24.42.101",
            "client_id":null,
            "mac_address":"52:54:00:47:20:4d"
         },
         {
            "lldp": [
              [1, "04112233aabbcc"],
              [5, "737730312d646973742d31622d623132"]
            ],
            "product":"0x0001",
            "vendor":"0x1af4",
            "name":"eth0",
            "has_carrier":true,
            "ipv4_address":"172.24.42.100",
            "client_id":null,
            "mac_address":"52:54:00:4e:3d:30"
         }
      ],
      "disks":[
         {
            "rotational":true,
            "vendor":"0x1af4",
            "name":"/dev/vda",
            "hctl":null,
            "wwn_vendor_extension":null,
            "wwn_with_extension":null,
            "model":"",
            "wwn":null,
            "serial":null,
            "size":13958643712
         }
      ],
      "boot":{
         "current_boot_mode":"bios",
         "pxe_interface":"52:54:00:4e:3d:30"
      },
      "system_vendor":{
         "serial_number":"Not Specified",
         "product_name":"Bochs",
         "manufacturer":"Bochs"
      },
      "memory":{
         "physical_mb":2048,
         "total":2105864192
      },
      "cpu":{
         "count":2,
         "frequency":"2100.084",
      "flags": [
        "fpu",
        "mmx",
        "fxsr",
        "sse",
        "sse2"
        ],
         "architecture":"x86_64"
      }
   },
   "error":null,
   "local_gb":12,
   "all_interfaces":{
      "eth1":{
         "ip":"172.24.42.101",
         "mac":"52:54:00:47:20:4d",
         "pxe":false,
         "client_id":null
      },
      "eth0":{
         "ip":"172.24.42.100",
         "mac":"52:54:00:4e:3d:30",
         "pxe":true,
         "client_id":null,
         "lldp_processed":{
           "switch_chassis_id":"11:22:33:aa:bb:cc",
           "switch_system_name":"sw01-dist-1b-b12"
         }
      }
   }
}
`

var (
	fooTimeStarted, _  = time.Parse(gophercloud.RFC3339NoZ, "2017-08-17T11:33:43")
	fooTimeFinished, _ = time.Parse(gophercloud.RFC3339NoZ, "2017-08-17T11:36:16")
	IntrospectionFoo   = introspection.Introspection{
		Finished:   true,
		State:      "finished",
		Error:      "",
		UUID:       "05ccda19-581b-49bf-8f5a-6ded99701d87",
		StartedAt:  fooTimeStarted,
		FinishedAt: fooTimeFinished,
		Links: []interface{}{
			map[string]interface{}{
				"href": "http://127.0.0.1:5050/v1/introspection/05ccda19-581b-49bf-8f5a-6ded99701d87",
				"rel":  "self",
			},
		},
	}

	barTimeStarted, _  = time.Parse(gophercloud.RFC3339NoZ, "2017-08-16T12:22:01")
	barTimeFinished, _ = time.Parse(gophercloud.RFC3339NoZ, "2017-08-16T12:24:30")
	IntrospectionBar   = introspection.Introspection{
		Finished:   true,
		State:      "finished",
		Error:      "",
		UUID:       "c244557e-899f-46fa-a1ff-5b2c6718616b",
		StartedAt:  barTimeStarted,
		FinishedAt: barTimeFinished,
		Links: []interface{}{
			map[string]interface{}{
				"href": "http://127.0.0.1:5050/v1/introspection/c244557e-899f-46fa-a1ff-5b2c6718616b",
				"rel":  "self",
			},
		},
	}

	IntrospectionDataRes = introspection.Data{
		CPUArch: "x86_64",
		MACs:    []string{"52:54:00:4e:3d:30"},
		RootDisk: introspection.RootDiskType{
			Rotational: true,
			Model:      "",
			Name:       "/dev/vda",
			Size:       13958643712,
			Vendor:     "0x1af4",
		},
		Interfaces: map[string]introspection.BaseInterfaceType{
			"eth0": {
				IP:  "172.24.42.100",
				MAC: "52:54:00:4e:3d:30",
				PXE: true,
			},
		},
		CPUs:          2,
		BootInterface: "52:54:00:4e:3d:30",
		MemoryMB:      2048,
		IPMIAddress:   "192.167.2.134",
		Inventory: introspection.InventoryType{
			SystemVendor: introspection.SystemVendorType{
				Manufacturer: "Bochs",
				ProductName:  "Bochs",
				SerialNumber: "Not Specified",
			},
			BmcAddress: "192.167.2.134",
			Boot: introspection.BootInfoType{
				CurrentBootMode: "bios",
				PXEInterface:    "52:54:00:4e:3d:30",
			},
			CPU: introspection.CPUType{
				Count:        2,
				Flags:        []string{"fpu", "mmx", "fxsr", "sse", "sse2"},
				Frequency:    "2100.084",
				Architecture: "x86_64",
			},
			Disks: []introspection.RootDiskType{
				introspection.RootDiskType{
					Rotational: true,
					Model:      "",
					Name:       "/dev/vda",
					Size:       13958643712,
					Vendor:     "0x1af4",
				},
			},
			Interfaces: []introspection.InterfaceType{
				introspection.InterfaceType{
					Vendor:      "0x1af4",
					HasCarrier:  true,
					MACAddress:  "52:54:00:47:20:4d",
					Name:        "eth1",
					Product:     "0x0001",
					IPV4Address: "172.24.42.101",
					LLDP:        []introspection.LLDPTLVType{},
				},
				introspection.InterfaceType{
					IPV4Address: "172.24.42.100",
					MACAddress:  "52:54:00:4e:3d:30",
					Name:        "eth0",
					Product:     "0x0001",
					HasCarrier:  true,
					Vendor:      "0x1af4",
					LLDP: []introspection.LLDPTLVType{
						introspection.LLDPTLVType{
							Type:  1,
							Value: "04112233aabbcc",
						},
						introspection.LLDPTLVType{
							Type:  5,
							Value: "737730312d646973742d31622d623132",
						},
					},
				},
			},
			Memory: introspection.MemoryType{
				PhysicalMb: 2048.0,
				Total:      2.105864192e+09,
			},
		},
		Error:   "",
		LocalGB: 12,
		AllInterfaces: map[string]introspection.BaseInterfaceType{
			"eth1": {
				IP:  "172.24.42.101",
				MAC: "52:54:00:47:20:4d",
				PXE: false,
			},
			"eth0": {
				IP:  "172.24.42.100",
				MAC: "52:54:00:4e:3d:30",
				PXE: true,
				LLDPProcessed: map[string]interface{}{
					"switch_chassis_id":  "11:22:33:aa:bb:cc",
					"switch_system_name": "sw01-dist-1b-b12",
				},
			},
		},
	}
)

// HandleListIntrospectionsSuccessfully sets up the test server to respond to a server ListIntrospections request.
func HandleListIntrospectionsSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/introspection", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		w.Header().Add("Content-Type", "application/json")
		r.ParseForm()

		marker := r.Form.Get("marker")

		switch marker {
		case "":
			fmt.Fprintf(w, IntrospectionListBody)

		case "c244557e-899f-46fa-a1ff-5b2c6718616b":
			fmt.Fprintf(w, `{ "introspection": [] }`)

		default:
			t.Fatalf("/introspection invoked with unexpected marker=[%s]", marker)
		}
	})
}

// HandleGetIntrospectionStatusSuccessfully sets up the test server to respond to a GetIntrospectionStatus request.
func HandleGetIntrospectionStatusSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/introspection/c244557e-899f-46fa-a1ff-5b2c6718616b", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")
		fmt.Fprintf(w, IntrospectionStatus)
	})
}

// HandleStartIntrospectionSuccessfully sets up the test server to respond to a StartIntrospection request.
func HandleStartIntrospectionSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/introspection/c244557e-899f-46fa-a1ff-5b2c6718616b", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		w.WriteHeader(http.StatusAccepted)
	})
}

// HandleAbortIntrospectionSuccessfully sets up the test server to respond to an AbortIntrospection request.
func HandleAbortIntrospectionSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/introspection/c244557e-899f-46fa-a1ff-5b2c6718616b/abort", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		w.WriteHeader(http.StatusAccepted)
	})
}

// HandleGetIntrospectionDataSuccessfully sets up the test server to respond to a GetIntrospectionData request.
func HandleGetIntrospectionDataSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/introspection/c244557e-899f-46fa-a1ff-5b2c6718616b/data", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		fmt.Fprintf(w, IntrospectionDataJSONSample)
	})
}

// HandleReApplyIntrospectionSuccessfully sets up the test server to respond to a ReApplyIntrospection request.
func HandleReApplyIntrospectionSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/introspection/c244557e-899f-46fa-a1ff-5b2c6718616b/data/unprocessed", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		w.WriteHeader(http.StatusAccepted)
	})
}
