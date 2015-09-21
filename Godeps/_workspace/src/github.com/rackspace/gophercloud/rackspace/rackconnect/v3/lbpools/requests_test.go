package lbpools

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	"github.com/rackspace/gophercloud/testhelper/client"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

func TestListPools(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	th.Mux.HandleFunc("/load_balancer_pools", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `[
      {
        "id": "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2",
        "name": "RCv3Test",
        "node_counts": {
          "cloud_servers": 3,
          "external": 4,
          "total": 7
        },
        "port": 80,
        "status": "ACTIVE",
        "status_detail": null,
        "virtual_ip": "203.0.113.5"
      },
      {
        "id": "33021100-4abf-4836-9080-465a6d87ab68",
        "name": "RCv3Test2",
        "node_counts": {
          "cloud_servers": 1,
          "external": 0,
          "total": 1
        },
        "port": 80,
        "status": "ACTIVE",
        "status_detail": null,
        "virtual_ip": "203.0.113.7"
      },
      {
        "id": "b644350a-301b-47b5-a411-c6e0f933c347",
        "name": "RCv3Test3",
        "node_counts": {
          "cloud_servers": 2,
          "external": 3,
          "total": 5
        },
        "port": 443,
        "status": "ACTIVE",
        "status_detail": null,
        "virtual_ip": "203.0.113.15"
      }
    ]`)
	})

	expected := []Pool{
		Pool{
			ID:   "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2",
			Name: "RCv3Test",
			NodeCounts: struct {
				CloudServers int `mapstructure:"cloud_servers"`
				External     int `mapstructure:"external"`
				Total        int `mapstructure:"total"`
			}{
				CloudServers: 3,
				External:     4,
				Total:        7,
			},
			Port:      80,
			Status:    "ACTIVE",
			VirtualIP: "203.0.113.5",
		},
		Pool{
			ID:   "33021100-4abf-4836-9080-465a6d87ab68",
			Name: "RCv3Test2",
			NodeCounts: struct {
				CloudServers int `mapstructure:"cloud_servers"`
				External     int `mapstructure:"external"`
				Total        int `mapstructure:"total"`
			}{
				CloudServers: 1,
				External:     0,
				Total:        1,
			},
			Port:      80,
			Status:    "ACTIVE",
			VirtualIP: "203.0.113.7",
		},
		Pool{
			ID:   "b644350a-301b-47b5-a411-c6e0f933c347",
			Name: "RCv3Test3",
			NodeCounts: struct {
				CloudServers int `mapstructure:"cloud_servers"`
				External     int `mapstructure:"external"`
				Total        int `mapstructure:"total"`
			}{
				CloudServers: 2,
				External:     3,
				Total:        5,
			},
			Port:      443,
			Status:    "ACTIVE",
			VirtualIP: "203.0.113.15",
		},
	}

	count := 0
	err := List(fake.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractPools(page)
		th.AssertNoErr(t, err)

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, count, 1)
}

func TestGetLBPool(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	th.Mux.HandleFunc("/load_balancer_pools/d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `{
      "id": "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2",
      "name": "RCv3Test",
      "node_counts": {
        "cloud_servers": 3,
        "external": 4,
        "total": 7
      },
      "port": 80,
      "status": "ACTIVE",
      "status_detail": null,
      "virtual_ip": "203.0.113.5"
    }`)
	})

	expected := &Pool{
		ID:   "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2",
		Name: "RCv3Test",
		NodeCounts: struct {
			CloudServers int `mapstructure:"cloud_servers"`
			External     int `mapstructure:"external"`
			Total        int `mapstructure:"total"`
		}{
			CloudServers: 3,
			External:     4,
			Total:        7,
		},
		Port:      80,
		Status:    "ACTIVE",
		VirtualIP: "203.0.113.5",
	}

	actual, err := Get(fake.ServiceClient(), "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2").Extract()
	th.AssertNoErr(t, err)

	th.AssertDeepEquals(t, expected, actual)
}

func TestListNodes(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	th.Mux.HandleFunc("/load_balancer_pools/d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2/nodes", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `[
      {
        "created": "2014-05-30T03:23:42Z",
        "cloud_server": {
          "id": "d95ae0c4-6ab8-4873-b82f-f8433840cff2"
        },
        "id": "1860451d-fb89-45b8-b54e-151afceb50e5",
        "load_balancer_pool": {
          "id": "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2"
        },
        "status": "ACTIVE",
        "updated": "2014-05-30T03:24:18Z"
      },
      {
        "created": "2014-05-31T08:23:12Z",
        "cloud_server": {
          "id": "f28b870f-a063-498a-8b12-7025e5b1caa6"
        },
        "id": "b70481dd-7edf-4dbb-a44b-41cc7679d4fb",
        "load_balancer_pool": {
          "id": "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2"
        },
        "status": "ADDING",
        "updated": "2014-05-31T08:23:26Z"
      },
      {
        "created": "2014-05-31T08:23:18Z",
        "cloud_server": {
          "id": "a3d3a6b3-e4e4-496f-9a3d-5c987163e458"
        },
        "id": "ced9ddc8-6fae-4e72-9457-16ead52b5515",
        "load_balancer_pool": {
          "id": "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2"
        },
        "status": "ADD_FAILED",
        "status_detail": "Unable to communicate with network device",
        "updated": "2014-05-31T08:24:36Z"
      }
    ]`)
	})

	expected := []Node{
		Node{
			CreatedAt: time.Date(2014, 5, 30, 3, 23, 42, 0, time.UTC),
			CloudServer: struct {
				ID string `mapstructure:"id"`
			}{
				ID: "d95ae0c4-6ab8-4873-b82f-f8433840cff2",
			},
			ID: "1860451d-fb89-45b8-b54e-151afceb50e5",
			LoadBalancerPool: struct {
				ID string `mapstructure:"id"`
			}{
				ID: "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2",
			},
			Status:    "ACTIVE",
			UpdatedAt: time.Date(2014, 5, 30, 3, 24, 18, 0, time.UTC),
		},
		Node{
			CreatedAt: time.Date(2014, 5, 31, 8, 23, 12, 0, time.UTC),
			CloudServer: struct {
				ID string `mapstructure:"id"`
			}{
				ID: "f28b870f-a063-498a-8b12-7025e5b1caa6",
			},
			ID: "b70481dd-7edf-4dbb-a44b-41cc7679d4fb",
			LoadBalancerPool: struct {
				ID string `mapstructure:"id"`
			}{
				ID: "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2",
			},
			Status:    "ADDING",
			UpdatedAt: time.Date(2014, 5, 31, 8, 23, 26, 0, time.UTC),
		},
		Node{
			CreatedAt: time.Date(2014, 5, 31, 8, 23, 18, 0, time.UTC),
			CloudServer: struct {
				ID string `mapstructure:"id"`
			}{
				ID: "a3d3a6b3-e4e4-496f-9a3d-5c987163e458",
			},
			ID: "ced9ddc8-6fae-4e72-9457-16ead52b5515",
			LoadBalancerPool: struct {
				ID string `mapstructure:"id"`
			}{
				ID: "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2",
			},
			Status:       "ADD_FAILED",
			StatusDetail: "Unable to communicate with network device",
			UpdatedAt:    time.Date(2014, 5, 31, 8, 24, 36, 0, time.UTC),
		},
	}

	count := 0
	err := ListNodes(fake.ServiceClient(), "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2").EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractNodes(page)
		th.AssertNoErr(t, err)

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, count, 1)
}

func TestCreateNode(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	th.Mux.HandleFunc("/load_balancer_pools/d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2/nodes", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
      {
        "cloud_server": {
          "id": "d95ae0c4-6ab8-4873-b82f-f8433840cff2"
        }
      }
    `)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		fmt.Fprintf(w, `
      {
        "created": "2014-05-30T03:23:42Z",
        "cloud_server": {
          "id": "d95ae0c4-6ab8-4873-b82f-f8433840cff2"
        },
        "id": "1860451d-fb89-45b8-b54e-151afceb50e5",
        "load_balancer_pool": {
          "id": "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2"
        },
        "status": "ACTIVE",
        "status_detail": null,
        "updated": "2014-05-30T03:24:18Z"
      }
    `)
	})

	expected := &Node{
		CreatedAt: time.Date(2014, 5, 30, 3, 23, 42, 0, time.UTC),
		CloudServer: struct {
			ID string `mapstructure:"id"`
		}{
			ID: "d95ae0c4-6ab8-4873-b82f-f8433840cff2",
		},
		ID: "1860451d-fb89-45b8-b54e-151afceb50e5",
		LoadBalancerPool: struct {
			ID string `mapstructure:"id"`
		}{
			ID: "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2",
		},
		Status:    "ACTIVE",
		UpdatedAt: time.Date(2014, 5, 30, 3, 24, 18, 0, time.UTC),
	}

	actual, err := CreateNode(fake.ServiceClient(), "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2", "d95ae0c4-6ab8-4873-b82f-f8433840cff2").Extract()
	th.AssertNoErr(t, err)

	th.AssertDeepEquals(t, expected, actual)
}

func TestListNodesDetails(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	th.Mux.HandleFunc("/load_balancer_pools/d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2/nodes/details", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `
      [
      {
        "created": "2014-05-30T03:23:42Z",
        "cloud_server": {
          "cloud_network": {
            "cidr": "192.168.100.0/24",
            "created": "2014-05-25T01:23:42Z",
            "id": "07426958-1ebf-4c38-b032-d456820ca21a",
            "name": "RC-CLOUD",
            "private_ip_v4": "192.168.100.5",
            "updated": "2014-05-25T02:28:44Z"
          },
          "created": "2014-05-30T02:18:42Z",
          "id": "d95ae0c4-6ab8-4873-b82f-f8433840cff2",
          "name": "RCv3TestServer1",
          "updated": "2014-05-30T02:19:18Z"
        },
        "id": "1860451d-fb89-45b8-b54e-151afceb50e5",
        "load_balancer_pool": {
          "id": "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2",
          "name": "RCv3Test",
          "node_counts": {
            "cloud_servers": 3,
            "external": 4,
            "total": 7
          },
          "port": 80,
          "status": "ACTIVE",
          "status_detail": null,
          "virtual_ip": "203.0.113.5"
        },
        "status": "ACTIVE",
        "status_detail": null,
        "updated": "2014-05-30T03:24:18Z"
      }
      ]
    `)
	})

	expected := []NodeDetails{
		NodeDetails{
			CreatedAt: time.Date(2014, 5, 30, 3, 23, 42, 0, time.UTC),
			CloudServer: struct {
				ID           string `mapstructure:"id"`
				Name         string `mapstructure:"name"`
				CloudNetwork struct {
					ID          string    `mapstructure:"id"`
					Name        string    `mapstructure:"name"`
					PrivateIPv4 string    `mapstructure:"private_ip_v4"`
					CIDR        string    `mapstructure:"cidr"`
					CreatedAt   time.Time `mapstructure:"-"`
					UpdatedAt   time.Time `mapstructure:"-"`
				} `mapstructure:"cloud_network"`
				CreatedAt time.Time `mapstructure:"-"`
				UpdatedAt time.Time `mapstructure:"-"`
			}{
				ID: "d95ae0c4-6ab8-4873-b82f-f8433840cff2",
				CloudNetwork: struct {
					ID          string    `mapstructure:"id"`
					Name        string    `mapstructure:"name"`
					PrivateIPv4 string    `mapstructure:"private_ip_v4"`
					CIDR        string    `mapstructure:"cidr"`
					CreatedAt   time.Time `mapstructure:"-"`
					UpdatedAt   time.Time `mapstructure:"-"`
				}{
					ID:          "07426958-1ebf-4c38-b032-d456820ca21a",
					CIDR:        "192.168.100.0/24",
					CreatedAt:   time.Date(2014, 5, 25, 1, 23, 42, 0, time.UTC),
					Name:        "RC-CLOUD",
					PrivateIPv4: "192.168.100.5",
					UpdatedAt:   time.Date(2014, 5, 25, 2, 28, 44, 0, time.UTC),
				},
				CreatedAt: time.Date(2014, 5, 30, 2, 18, 42, 0, time.UTC),
				Name:      "RCv3TestServer1",
				UpdatedAt: time.Date(2014, 5, 30, 2, 19, 18, 0, time.UTC),
			},
			ID: "1860451d-fb89-45b8-b54e-151afceb50e5",
			LoadBalancerPool: Pool{
				ID:   "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2",
				Name: "RCv3Test",
				NodeCounts: struct {
					CloudServers int `mapstructure:"cloud_servers"`
					External     int `mapstructure:"external"`
					Total        int `mapstructure:"total"`
				}{
					CloudServers: 3,
					External:     4,
					Total:        7,
				},
				Port:      80,
				Status:    "ACTIVE",
				VirtualIP: "203.0.113.5",
			},
			Status:    "ACTIVE",
			UpdatedAt: time.Date(2014, 5, 30, 3, 24, 18, 0, time.UTC),
		},
	}
	count := 0
	err := ListNodesDetails(fake.ServiceClient(), "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2").EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractNodesDetails(page)
		th.AssertNoErr(t, err)

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, count, 1)
}

func TestGetNode(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	th.Mux.HandleFunc("/load_balancer_pools/d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2/nodes/1860451d-fb89-45b8-b54e-151afceb50e5", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `
      {
        "created": "2014-05-30T03:23:42Z",
        "cloud_server": {
          "id": "d95ae0c4-6ab8-4873-b82f-f8433840cff2"
        },
        "id": "1860451d-fb89-45b8-b54e-151afceb50e5",
        "load_balancer_pool": {
          "id": "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2"
        },
        "status": "ACTIVE",
        "status_detail": null,
        "updated": "2014-05-30T03:24:18Z"
      }
    `)
	})

	expected := &Node{
		CreatedAt: time.Date(2014, 5, 30, 3, 23, 42, 0, time.UTC),
		CloudServer: struct {
			ID string `mapstructure:"id"`
		}{
			ID: "d95ae0c4-6ab8-4873-b82f-f8433840cff2",
		},
		ID: "1860451d-fb89-45b8-b54e-151afceb50e5",
		LoadBalancerPool: struct {
			ID string `mapstructure:"id"`
		}{
			ID: "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2",
		},
		Status:    "ACTIVE",
		UpdatedAt: time.Date(2014, 5, 30, 3, 24, 18, 0, time.UTC),
	}

	actual, err := GetNode(fake.ServiceClient(), "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2", "1860451d-fb89-45b8-b54e-151afceb50e5").Extract()
	th.AssertNoErr(t, err)

	th.AssertDeepEquals(t, expected, actual)
}

func TestDeleteNode(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	th.Mux.HandleFunc("/load_balancer_pools/d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2/nodes/1860451d-fb89-45b8-b54e-151afceb50e5", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNoContent)
	})

	err := DeleteNode(client.ServiceClient(), "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2", "1860451d-fb89-45b8-b54e-151afceb50e5").ExtractErr()
	th.AssertNoErr(t, err)
}

func TestGetNodeDetails(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	th.Mux.HandleFunc("/load_balancer_pools/d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2/nodes/d95ae0c4-6ab8-4873-b82f-f8433840cff2/details", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `
      {
        "created": "2014-05-30T03:23:42Z",
        "cloud_server": {
          "cloud_network": {
            "cidr": "192.168.100.0/24",
            "created": "2014-05-25T01:23:42Z",
            "id": "07426958-1ebf-4c38-b032-d456820ca21a",
            "name": "RC-CLOUD",
            "private_ip_v4": "192.168.100.5",
            "updated": "2014-05-25T02:28:44Z"
          },
          "created": "2014-05-30T02:18:42Z",
          "id": "d95ae0c4-6ab8-4873-b82f-f8433840cff2",
          "name": "RCv3TestServer1",
          "updated": "2014-05-30T02:19:18Z"
        },
        "id": "1860451d-fb89-45b8-b54e-151afceb50e5",
        "load_balancer_pool": {
          "id": "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2",
          "name": "RCv3Test",
          "node_counts": {
            "cloud_servers": 3,
            "external": 4,
            "total": 7
          },
          "port": 80,
          "status": "ACTIVE",
          "status_detail": null,
          "virtual_ip": "203.0.113.5"
        },
        "status": "ACTIVE",
        "status_detail": null,
        "updated": "2014-05-30T03:24:18Z"
      }
    `)
	})

	expected := &NodeDetails{
		CreatedAt: time.Date(2014, 5, 30, 3, 23, 42, 0, time.UTC),
		CloudServer: struct {
			ID           string `mapstructure:"id"`
			Name         string `mapstructure:"name"`
			CloudNetwork struct {
				ID          string    `mapstructure:"id"`
				Name        string    `mapstructure:"name"`
				PrivateIPv4 string    `mapstructure:"private_ip_v4"`
				CIDR        string    `mapstructure:"cidr"`
				CreatedAt   time.Time `mapstructure:"-"`
				UpdatedAt   time.Time `mapstructure:"-"`
			} `mapstructure:"cloud_network"`
			CreatedAt time.Time `mapstructure:"-"`
			UpdatedAt time.Time `mapstructure:"-"`
		}{
			ID: "d95ae0c4-6ab8-4873-b82f-f8433840cff2",
			CloudNetwork: struct {
				ID          string    `mapstructure:"id"`
				Name        string    `mapstructure:"name"`
				PrivateIPv4 string    `mapstructure:"private_ip_v4"`
				CIDR        string    `mapstructure:"cidr"`
				CreatedAt   time.Time `mapstructure:"-"`
				UpdatedAt   time.Time `mapstructure:"-"`
			}{
				ID:          "07426958-1ebf-4c38-b032-d456820ca21a",
				CIDR:        "192.168.100.0/24",
				CreatedAt:   time.Date(2014, 5, 25, 1, 23, 42, 0, time.UTC),
				Name:        "RC-CLOUD",
				PrivateIPv4: "192.168.100.5",
				UpdatedAt:   time.Date(2014, 5, 25, 2, 28, 44, 0, time.UTC),
			},
			CreatedAt: time.Date(2014, 5, 30, 2, 18, 42, 0, time.UTC),
			Name:      "RCv3TestServer1",
			UpdatedAt: time.Date(2014, 5, 30, 2, 19, 18, 0, time.UTC),
		},
		ID: "1860451d-fb89-45b8-b54e-151afceb50e5",
		LoadBalancerPool: Pool{
			ID:   "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2",
			Name: "RCv3Test",
			NodeCounts: struct {
				CloudServers int `mapstructure:"cloud_servers"`
				External     int `mapstructure:"external"`
				Total        int `mapstructure:"total"`
			}{
				CloudServers: 3,
				External:     4,
				Total:        7,
			},
			Port:      80,
			Status:    "ACTIVE",
			VirtualIP: "203.0.113.5",
		},
		Status:    "ACTIVE",
		UpdatedAt: time.Date(2014, 5, 30, 3, 24, 18, 0, time.UTC),
	}

	actual, err := GetNodeDetails(fake.ServiceClient(), "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2", "d95ae0c4-6ab8-4873-b82f-f8433840cff2").Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, expected, actual)
}

func TestCreateNodes(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	th.Mux.HandleFunc("/load_balancer_pools/nodes", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
      [
      {
        "cloud_server": {
          "id": "d95ae0c4-6ab8-4873-b82f-f8433840cff2"
        },
        "load_balancer_pool": {
          "id": "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2"
        }
      },
      {
        "cloud_server": {
          "id": "d95ae0c4-6ab8-4873-b82f-f8433840cff2"
        },
        "load_balancer_pool": {
          "id": "33021100-4abf-4836-9080-465a6d87ab68"
      }
    }
    ]
  `)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		fmt.Fprintf(w, `
      [
      {
        "created": "2014-05-30T03:23:42Z",
        "cloud_server": {
          "id": "d95ae0c4-6ab8-4873-b82f-f8433840cff2"
        },
        "id": "1860451d-fb89-45b8-b54e-151afceb50e5",
        "load_balancer_pool": {
          "id": "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2"
        },
        "status": "ADDING",
        "status_detail": null,
        "updated": null
      },
      {
        "created": "2014-05-31T08:23:12Z",
        "cloud_server": {
          "id": "d95ae0c4-6ab8-4873-b82f-f8433840cff2"
        },
        "id": "b70481dd-7edf-4dbb-a44b-41cc7679d4fb",
        "load_balancer_pool": {
          "id": "33021100-4abf-4836-9080-465a6d87ab68"
        },
        "status": "ADDING",
        "status_detail": null,
        "updated": null
      }
      ]
    `)
	})

	expected := []Node{
		Node{
			CreatedAt: time.Date(2014, 5, 30, 3, 23, 42, 0, time.UTC),
			CloudServer: struct {
				ID string `mapstructure:"id"`
			}{
				ID: "d95ae0c4-6ab8-4873-b82f-f8433840cff2",
			},
			ID: "1860451d-fb89-45b8-b54e-151afceb50e5",
			LoadBalancerPool: struct {
				ID string `mapstructure:"id"`
			}{
				ID: "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2",
			},
			Status: "ADDING",
		},
		Node{
			CreatedAt: time.Date(2014, 5, 31, 8, 23, 12, 0, time.UTC),
			CloudServer: struct {
				ID string `mapstructure:"id"`
			}{
				ID: "d95ae0c4-6ab8-4873-b82f-f8433840cff2",
			},
			ID: "b70481dd-7edf-4dbb-a44b-41cc7679d4fb",
			LoadBalancerPool: struct {
				ID string `mapstructure:"id"`
			}{
				ID: "33021100-4abf-4836-9080-465a6d87ab68",
			},
			Status: "ADDING",
		},
	}

	opts := NodesOpts{
		NodeOpts{
			ServerID: "d95ae0c4-6ab8-4873-b82f-f8433840cff2",
			PoolID:   "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2",
		},
		NodeOpts{
			ServerID: "d95ae0c4-6ab8-4873-b82f-f8433840cff2",
			PoolID:   "33021100-4abf-4836-9080-465a6d87ab68",
		},
	}
	actual, err := CreateNodes(fake.ServiceClient(), opts).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, expected, actual)
}

func TestDeleteNodes(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	th.Mux.HandleFunc("/load_balancer_pools/nodes", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
      [
      {
        "cloud_server": {
          "id": "d95ae0c4-6ab8-4873-b82f-f8433840cff2"
        },
        "load_balancer_pool": {
          "id": "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2"
        }
      },
      {
        "cloud_server": {
          "id": "d95ae0c4-6ab8-4873-b82f-f8433840cff2"
        },
        "load_balancer_pool": {
          "id": "33021100-4abf-4836-9080-465a6d87ab68"
        }
      }
      ]
    `)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNoContent)
	})

	opts := NodesOpts{
		NodeOpts{
			ServerID: "d95ae0c4-6ab8-4873-b82f-f8433840cff2",
			PoolID:   "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2",
		},
		NodeOpts{
			ServerID: "d95ae0c4-6ab8-4873-b82f-f8433840cff2",
			PoolID:   "33021100-4abf-4836-9080-465a6d87ab68",
		},
	}
	err := DeleteNodes(client.ServiceClient(), opts).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestListNodesForServerDetails(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	th.Mux.HandleFunc("/load_balancer_pools/nodes/details", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `
      [
      {
        "created": "2014-05-30T03:23:42Z",
        "id": "1860451d-fb89-45b8-b54e-151afceb50e5",
        "load_balancer_pool": {
          "id": "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2",
          "name": "RCv3Test",
          "node_counts": {
            "cloud_servers": 3,
            "external": 4,
            "total": 7
          },
          "port": 80,
          "status": "ACTIVE",
          "status_detail": null,
          "virtual_ip": "203.0.113.5"
        },
        "status": "ACTIVE",
        "status_detail": null,
        "updated": "2014-05-30T03:24:18Z"
      }
      ]
    `)
	})

	expected := []NodeDetailsForServer{
		NodeDetailsForServer{
			CreatedAt: time.Date(2014, 5, 30, 3, 23, 42, 0, time.UTC),
			ID:        "1860451d-fb89-45b8-b54e-151afceb50e5",
			LoadBalancerPool: Pool{
				ID:   "d6d3aa7c-dfa5-4e61-96ee-1d54ac1075d2",
				Name: "RCv3Test",
				NodeCounts: struct {
					CloudServers int `mapstructure:"cloud_servers"`
					External     int `mapstructure:"external"`
					Total        int `mapstructure:"total"`
				}{
					CloudServers: 3,
					External:     4,
					Total:        7,
				},
				Port:      80,
				Status:    "ACTIVE",
				VirtualIP: "203.0.113.5",
			},
			Status:    "ACTIVE",
			UpdatedAt: time.Date(2014, 5, 30, 3, 24, 18, 0, time.UTC),
		},
	}
	count := 0
	err := ListNodesDetailsForServer(fake.ServiceClient(), "07426958-1ebf-4c38-b032-d456820ca21a").EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractNodesDetailsForServer(page)
		th.AssertNoErr(t, err)

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, count, 1)
}
