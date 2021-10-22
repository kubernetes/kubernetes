package testing

import (
	"fmt"
	"net/http"
	"testing"

	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

func createReq(name, description, network, subnetwork string) string {
	return fmt.Sprintf(`{
        "share_network": {
            "name": "%s",
            "description": "%s",
            "neutron_net_id": "%s",
            "neutron_subnet_id": "%s"
        }
    }`, name, description, network, subnetwork)
}

func createResp(name, description, network, subnetwork string) string {
	return fmt.Sprintf(`
    {
        "share_network": {
            "name": "%s",
            "description": "%s",
            "segmentation_id": null,
            "created_at": "2015-09-07T14:37:00.583656",
            "updated_at": null,
            "id": "77eb3421-4549-4789-ac39-0d5185d68c29",
            "neutron_net_id": "%s",
            "neutron_subnet_id": "%s",
            "ip_version": null,
            "nova_net_id": null,
            "cidr": null,
            "project_id": "e10a683c20da41248cfd5e1ab3d88c62",
            "network_type": null
        }
    }`, name, description, network, subnetwork)
}

func MockCreateResponse(t *testing.T) {
	th.Mux.HandleFunc("/share-networks", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, createReq("my_network",
			"This is my share network",
			"998b42ee-2cee-4d36-8b95-67b5ca1f2109",
			"53482b62-2c84-4a53-b6ab-30d9d9800d06"))

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusAccepted)

		fmt.Fprintf(w, createResp("my_network",
			"This is my share network",
			"998b42ee-2cee-4d36-8b95-67b5ca1f2109",
			"53482b62-2c84-4a53-b6ab-30d9d9800d06"))
	})
}

func MockDeleteResponse(t *testing.T) {
	th.Mux.HandleFunc("/share-networks/fa158a3d-6d9f-4187-9ca5-abbb82646eb2", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusAccepted)
	})
}

func MockListResponse(t *testing.T) {
	th.Mux.HandleFunc("/share-networks/detail", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		r.ParseForm()
		marker := r.Form.Get("offset")

		switch marker {
		case "":
			fmt.Fprintf(w, `{
            "share_networks": [
                {
                    "name": "net_my1",
                    "segmentation_id": null,
                    "created_at": "2015-09-04T14:57:13.000000",
                    "neutron_subnet_id": "53482b62-2c84-4a53-b6ab-30d9d9800d06",
                    "updated_at": null,
                    "id": "32763294-e3d4-456a-998d-60047677c2fb",
                    "neutron_net_id": "998b42ee-2cee-4d36-8b95-67b5ca1f2109",
                    "ip_version": null,
                    "nova_net_id": null,
                    "cidr": null,
                    "project_id": "16e1ab15c35a457e9c2b2aa189f544e1",
                    "network_type": null,
                    "description": "descr"
                },
                {
                    "name": "net_my",
                    "segmentation_id": null,
                    "created_at": "2015-09-04T14:54:25.000000",
                    "neutron_subnet_id": "53482b62-2c84-4a53-b6ab-30d9d9800d06",
                    "updated_at": null,
                    "id": "713df749-aac0-4a54-af52-10f6c991e80c",
                    "neutron_net_id": "998b42ee-2cee-4d36-8b95-67b5ca1f2109",
                    "ip_version": null,
                    "nova_net_id": null,
                    "cidr": null,
                    "project_id": "16e1ab15c35a457e9c2b2aa189f544e1",
                    "network_type": null,
                    "description": "desecr"
                },
                {
                    "name": null,
                    "segmentation_id": null,
                    "created_at": "2015-09-04T14:51:41.000000",
                    "neutron_subnet_id": null,
                    "updated_at": null,
                    "id": "fa158a3d-6d9f-4187-9ca5-abbb82646eb2",
                    "neutron_net_id": null,
                    "ip_version": null,
                    "nova_net_id": null,
                    "cidr": null,
                    "project_id": "16e1ab15c35a457e9c2b2aa189f544e1",
                    "network_type": null,
                    "description": null
                }
            ]
        }`)
		default:
			fmt.Fprintf(w, `
				{
					"share_networks": []
				}`)
		}
	})
}

func MockFilteredListResponse(t *testing.T) {
	th.Mux.HandleFunc("/share-networks/detail", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		r.ParseForm()
		marker := r.Form.Get("offset")
		switch marker {
		case "":
			fmt.Fprintf(w, `
				{
					"share_networks": [
						{
							"name": "net_my1",
							"segmentation_id": null,
							"created_at": "2015-09-04T14:57:13.000000",
							"neutron_subnet_id": "53482b62-2c84-4a53-b6ab-30d9d9800d06",
							"updated_at": null,
							"id": "32763294-e3d4-456a-998d-60047677c2fb",
							"neutron_net_id": "998b42ee-2cee-4d36-8b95-67b5ca1f2109",
							"ip_version": null,
							"nova_net_id": null,
							"cidr": null,
							"project_id": "16e1ab15c35a457e9c2b2aa189f544e1",
							"network_type": null,
							"description": "descr"
						}
					]
				}`)
		case "1":
			fmt.Fprintf(w, `
				{
					"share_networks": [
						{
							"name": "net_my1",
							"segmentation_id": null,
							"created_at": "2015-09-04T14:57:13.000000",
							"neutron_subnet_id": "53482b62-2c84-4a53-b6ab-30d9d9800d06",
							"updated_at": null,
							"id": "32763294-e3d4-456a-998d-60047677c2fb",
							"neutron_net_id": "998b42ee-2cee-4d36-8b95-67b5ca1f2109",
							"ip_version": null,
							"nova_net_id": null,
							"cidr": null,
							"project_id": "16e1ab15c35a457e9c2b2aa189f544e1",
							"network_type": null,
							"description": "descr"
						}
					]
				}`)
		case "2":
			fmt.Fprintf(w, `
				{
					"share_networks": [
						{
							"name": "net_my1",
							"segmentation_id": null,
							"created_at": "2015-09-04T14:57:13.000000",
							"neutron_subnet_id": "53482b62-2c84-4a53-b6ab-30d9d9800d06",
							"updated_at": null,
							"id": "32763294-e3d4-456a-998d-60047677c2fb",
							"neutron_net_id": "998b42ee-2cee-4d36-8b95-67b5ca1f2109",
							"ip_version": null,
							"nova_net_id": null,
							"cidr": null,
							"project_id": "16e1ab15c35a457e9c2b2aa189f544e1",
							"network_type": null,
							"description": "descr"
						}
					]
				}`)
		default:
			fmt.Fprintf(w, `
				{
					"share_networks": []
				}`)
		}
	})
}

func MockGetResponse(t *testing.T) {
	th.Mux.HandleFunc("/share-networks/7f950b52-6141-4a08-bbb5-bb7ffa3ea5fd", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `
        {
            "share_network": {
                "name": "net_my1",
                "segmentation_id": null,
                "created_at": "2015-09-04T14:56:45.000000",
                "neutron_subnet_id": "53482b62-2c84-4a53-b6ab-30d9d9800d06",
                "updated_at": null,
                "id": "7f950b52-6141-4a08-bbb5-bb7ffa3ea5fd",
                "neutron_net_id": "998b42ee-2cee-4d36-8b95-67b5ca1f2109",
                "ip_version": null,
                "nova_net_id": null,
                "cidr": null,
                "project_id": "16e1ab15c35a457e9c2b2aa189f544e1",
                "network_type": null,
                "description": "descr"
            }
        }`)
	})
}

func MockUpdateNeutronResponse(t *testing.T) {
	th.Mux.HandleFunc("/share-networks/713df749-aac0-4a54-af52-10f6c991e80c", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `
            {
                "share_network": {
                    "name": "net_my2",
                    "segmentation_id": null,
                    "created_at": "2015-09-04T14:54:25.000000",
                    "neutron_subnet_id": "new-neutron-subnet-id",
                    "updated_at": "2015-09-07T08:02:53.512184",
                    "id": "713df749-aac0-4a54-af52-10f6c991e80c",
                    "neutron_net_id": "new-neutron-id",
                    "ip_version": 4,
                    "nova_net_id": null,
                    "cidr": null,
                    "project_id": "16e1ab15c35a457e9c2b2aa189f544e1",
                    "network_type": null,
                    "description": "new description"
                }
            }
        `)
	})
}

func MockUpdateNovaResponse(t *testing.T) {
	th.Mux.HandleFunc("/share-networks/713df749-aac0-4a54-af52-10f6c991e80c", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `
            {
                "share_network": {
                    "name": "net_my2",
                    "segmentation_id": null,
                    "created_at": "2015-09-04T14:54:25.000000",
                    "neutron_subnet_id": null,
                    "updated_at": "2015-09-07T08:02:53.512184",
                    "id": "713df749-aac0-4a54-af52-10f6c991e80c",
                    "neutron_net_id": null,
                    "ip_version": 4,
                    "nova_net_id": "new-nova-id",
                    "cidr": null,
                    "project_id": "16e1ab15c35a457e9c2b2aa189f544e1",
                    "network_type": null,
                    "description": "new description"
                }
            }
        `)
	})
}

func MockAddSecurityServiceResponse(t *testing.T) {
	th.Mux.HandleFunc("/share-networks/shareNetworkID/action", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `
        {
            "share_network": {
                "name": "net2",
                "segmentation_id": null,
                "created_at": "2015-09-07T12:31:12.000000",
                "neutron_subnet_id": null,
                "updated_at": null,
                "id": "d8ae6799-2567-4a89-aafb-fa4424350d2b",
                "neutron_net_id": null,
                "ip_version": 4,
                "nova_net_id": "998b42ee-2cee-4d36-8b95-67b5ca1f2109",
                "cidr": null,
                "project_id": "16e1ab15c35a457e9c2b2aa189f544e1",
                "network_type": null,
                "description": null
            }
        }`)
	})
}

func MockRemoveSecurityServiceResponse(t *testing.T) {
	th.Mux.HandleFunc("/share-networks/shareNetworkID/action", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `
        {
            "share_network": {
                "name": "net2",
                "segmentation_id": null,
                "created_at": "2015-09-07T12:31:12.000000",
                "neutron_subnet_id": null,
                "updated_at": null,
                "id": "d8ae6799-2567-4a89-aafb-fa4424350d2b",
                "neutron_net_id": null,
                "ip_version": null,
                "nova_net_id": "998b42ee-2cee-4d36-8b95-67b5ca1f2109",
                "cidr": null,
                "project_id": "16e1ab15c35a457e9c2b2aa189f544e1",
                "network_type": null,
                "description": null
            }
        }`)
	})
}
