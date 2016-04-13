package publicips

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

func TestListIPs(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	th.Mux.HandleFunc("/public_ips", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `[
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
        "id": "2d0f586b-37a7-4ae0-adac-2743d5feb450",
        "public_ip_v4": "203.0.113.110",
        "status": "ACTIVE",
        "status_detail": null,
        "updated": "2014-05-30T03:24:18Z"
      }
    ]`)
	})

	expected := []PublicIP{
		PublicIP{
			ID:         "2d0f586b-37a7-4ae0-adac-2743d5feb450",
			PublicIPv4: "203.0.113.110",
			CreatedAt:  time.Date(2014, 5, 30, 3, 23, 42, 0, time.UTC),
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
			Status:    "ACTIVE",
			UpdatedAt: time.Date(2014, 5, 30, 3, 24, 18, 0, time.UTC),
		},
	}

	count := 0
	err := List(fake.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractPublicIPs(page)
		th.AssertNoErr(t, err)

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, count, 1)
}

func TestCreateIP(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	th.Mux.HandleFunc("/public_ips", func(w http.ResponseWriter, r *http.Request) {
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
        "id": "2d0f586b-37a7-4ae0-adac-2743d5feb450",
        "status": "ADDING"
      }`)
	})

	expected := &PublicIP{
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
		ID:     "2d0f586b-37a7-4ae0-adac-2743d5feb450",
		Status: "ADDING",
	}

	actual, err := Create(fake.ServiceClient(), "d95ae0c4-6ab8-4873-b82f-f8433840cff2").Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, expected, actual)
}

func TestGetIP(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	th.Mux.HandleFunc("/public_ips/2d0f586b-37a7-4ae0-adac-2743d5feb450", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
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
        "id": "2d0f586b-37a7-4ae0-adac-2743d5feb450",
        "public_ip_v4": "203.0.113.110",
        "status": "ACTIVE",
        "status_detail": null,
        "updated": "2014-05-30T03:24:18Z"
      }`)
	})

	expected := &PublicIP{
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
		ID:         "2d0f586b-37a7-4ae0-adac-2743d5feb450",
		Status:     "ACTIVE",
		PublicIPv4: "203.0.113.110",
		UpdatedAt:  time.Date(2014, 5, 30, 3, 24, 18, 0, time.UTC),
	}

	actual, err := Get(fake.ServiceClient(), "2d0f586b-37a7-4ae0-adac-2743d5feb450").Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, expected, actual)
}

func TestDeleteIP(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	th.Mux.HandleFunc("/public_ips/2d0f586b-37a7-4ae0-adac-2743d5feb450", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNoContent)
	})

	err := Delete(client.ServiceClient(), "2d0f586b-37a7-4ae0-adac-2743d5feb450").ExtractErr()
	th.AssertNoErr(t, err)
}

func TestListForServer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	th.Mux.HandleFunc("/public_ips", func(w http.ResponseWriter, r *http.Request) {
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
      "id": "2d0f586b-37a7-4ae0-adac-2743d5feb450",
      "public_ip_v4": "203.0.113.110",
      "status": "ACTIVE",
      "updated": "2014-05-30T03:24:18Z"
    }
    ]`)
	})

	expected := []PublicIP{
		PublicIP{
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
			ID:         "2d0f586b-37a7-4ae0-adac-2743d5feb450",
			Status:     "ACTIVE",
			PublicIPv4: "203.0.113.110",
			UpdatedAt:  time.Date(2014, 5, 30, 3, 24, 18, 0, time.UTC),
		},
	}
	count := 0
	err := ListForServer(fake.ServiceClient(), "d95ae0c4-6ab8-4873-b82f-f8433840cff2").EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractPublicIPs(page)
		th.AssertNoErr(t, err)
		th.CheckDeepEquals(t, expected, actual)
		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, count, 1)
}
