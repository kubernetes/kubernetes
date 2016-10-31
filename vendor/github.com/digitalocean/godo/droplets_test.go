package godo

import (
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"testing"
)

func TestDroplets_ListDroplets(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/droplets", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"droplets": [{"id":1},{"id":2}]}`)
	})

	droplets, _, err := client.Droplets.List(nil)
	if err != nil {
		t.Errorf("Droplets.List returned error: %v", err)
	}

	expected := []Droplet{{ID: 1}, {ID: 2}}
	if !reflect.DeepEqual(droplets, expected) {
		t.Errorf("Droplets.List\n got=%#v\nwant=%#v", droplets, expected)
	}
}

func TestDroplets_ListDropletsByTag(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/droplets", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("tag_name") != "testing-1" {
			t.Errorf("Droplets.ListByTag did not request with a tag parameter")
		}

		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"droplets": [{"id":1},{"id":2}]}`)
	})

	droplets, _, err := client.Droplets.ListByTag("testing-1", nil)
	if err != nil {
		t.Errorf("Droplets.ListByTag returned error: %v", err)
	}

	expected := []Droplet{{ID: 1}, {ID: 2}}
	if !reflect.DeepEqual(droplets, expected) {
		t.Errorf("Droplets.ListByTag returned %+v, expected %+v", droplets, expected)
	}
}

func TestDroplets_ListDropletsMultiplePages(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/droplets", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")

		dr := dropletsRoot{
			Droplets: []Droplet{
				{ID: 1},
				{ID: 2},
			},
			Links: &Links{
				Pages: &Pages{Next: "http://example.com/v2/droplets/?page=2"},
			},
		}

		b, err := json.Marshal(dr)
		if err != nil {
			t.Fatal(err)
		}

		fmt.Fprint(w, string(b))
	})

	_, resp, err := client.Droplets.List(nil)
	if err != nil {
		t.Fatal(err)
	}

	checkCurrentPage(t, resp, 1)
}

func TestDroplets_RetrievePageByNumber(t *testing.T) {
	setup()
	defer teardown()

	jBlob := `
	{
		"droplets": [{"id":1},{"id":2}],
		"links":{
			"pages":{
				"next":"http://example.com/v2/droplets/?page=3",
				"prev":"http://example.com/v2/droplets/?page=1",
				"last":"http://example.com/v2/droplets/?page=3",
				"first":"http://example.com/v2/droplets/?page=1"
			}
		}
	}`

	mux.HandleFunc("/v2/droplets", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, jBlob)
	})

	opt := &ListOptions{Page: 2}
	_, resp, err := client.Droplets.List(opt)
	if err != nil {
		t.Fatal(err)
	}

	checkCurrentPage(t, resp, 2)
}

func TestDroplets_GetDroplet(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/droplets/12345", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"droplet":{"id":12345}}`)
	})

	droplets, _, err := client.Droplets.Get(12345)
	if err != nil {
		t.Errorf("Droplet.Get returned error: %v", err)
	}

	expected := &Droplet{ID: 12345}
	if !reflect.DeepEqual(droplets, expected) {
		t.Errorf("Droplets.Get\n got=%#v\nwant=%#v", droplets, expected)
	}
}

func TestDroplets_Create(t *testing.T) {
	setup()
	defer teardown()

	createRequest := &DropletCreateRequest{
		Name:   "name",
		Region: "region",
		Size:   "size",
		Image: DropletCreateImage{
			ID: 1,
		},
		Volumes: []DropletCreateVolume{
			{Name: "hello-im-a-volume"},
			{ID: "hello-im-another-volume"},
			{Name: "hello-im-still-a-volume", ID: "should be ignored due to Name"},
		},
		Tags: []string{"one", "two"},
	}

	mux.HandleFunc("/v2/droplets", func(w http.ResponseWriter, r *http.Request) {
		expected := map[string]interface{}{
			"name":               "name",
			"region":             "region",
			"size":               "size",
			"image":              float64(1),
			"ssh_keys":           nil,
			"backups":            false,
			"ipv6":               false,
			"private_networking": false,
			"volumes": []interface{}{
				map[string]interface{}{"name": "hello-im-a-volume"},
				map[string]interface{}{"id": "hello-im-another-volume"},
				map[string]interface{}{"name": "hello-im-still-a-volume"},
			},
			"tags": []interface{}{"one", "two"},
		}

		var v map[string]interface{}
		err := json.NewDecoder(r.Body).Decode(&v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		if !reflect.DeepEqual(v, expected) {
			t.Errorf("Request body\n got=%#v\nwant=%#v", v, expected)
		}

		fmt.Fprintf(w, `{"droplet":{"id":1}, "links":{"actions": [{"id": 1, "href": "http://example.com", "rel": "create"}]}}`)
	})

	droplet, resp, err := client.Droplets.Create(createRequest)
	if err != nil {
		t.Errorf("Droplets.Create returned error: %v", err)
	}

	if id := droplet.ID; id != 1 {
		t.Errorf("expected id '%d', received '%d'", 1, id)
	}

	if a := resp.Links.Actions[0]; a.ID != 1 {
		t.Errorf("expected action id '%d', received '%d'", 1, a.ID)
	}
}

func TestDroplets_CreateMultiple(t *testing.T) {
	setup()
	defer teardown()

	createRequest := &DropletMultiCreateRequest{
		Names:  []string{"name1", "name2"},
		Region: "region",
		Size:   "size",
		Image: DropletCreateImage{
			ID: 1,
		},
		Tags: []string{"one", "two"},
	}

	mux.HandleFunc("/v2/droplets", func(w http.ResponseWriter, r *http.Request) {
		expected := map[string]interface{}{
			"names":              []interface{}{"name1", "name2"},
			"region":             "region",
			"size":               "size",
			"image":              float64(1),
			"ssh_keys":           nil,
			"backups":            false,
			"ipv6":               false,
			"private_networking": false,
			"tags":               []interface{}{"one", "two"},
		}

		var v map[string]interface{}
		err := json.NewDecoder(r.Body).Decode(&v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		if !reflect.DeepEqual(v, expected) {
			t.Errorf("Request body = %#v, expected %#v", v, expected)
		}

		fmt.Fprintf(w, `{"droplets":[{"id":1},{"id":2}], "links":{"actions": [{"id": 1, "href": "http://example.com", "rel": "multiple_create"}]}}`)
	})

	droplets, resp, err := client.Droplets.CreateMultiple(createRequest)
	if err != nil {
		t.Errorf("Droplets.CreateMultiple returned error: %v", err)
	}

	if id := droplets[0].ID; id != 1 {
		t.Errorf("expected id '%d', received '%d'", 1, id)
	}

	if id := droplets[1].ID; id != 2 {
		t.Errorf("expected id '%d', received '%d'", 1, id)
	}

	if a := resp.Links.Actions[0]; a.ID != 1 {
		t.Errorf("expected action id '%d', received '%d'", 1, a.ID)
	}
}

func TestDroplets_Destroy(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/droplets/12345", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "DELETE")
	})

	_, err := client.Droplets.Delete(12345)
	if err != nil {
		t.Errorf("Droplet.Delete returned error: %v", err)
	}
}

func TestDroplets_DestroyByTag(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/droplets", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("tag_name") != "testing-1" {
			t.Errorf("Droplets.DeleteByTag did not request with a tag parameter")
		}

		testMethod(t, r, "DELETE")
	})

	_, err := client.Droplets.DeleteByTag("testing-1")
	if err != nil {
		t.Errorf("Droplet.Delete returned error: %v", err)
	}
}

func TestDroplets_Kernels(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/droplets/12345/kernels", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"kernels": [{"id":1},{"id":2}]}`)
	})

	opt := &ListOptions{Page: 2}
	kernels, _, err := client.Droplets.Kernels(12345, opt)
	if err != nil {
		t.Errorf("Droplets.Kernels returned error: %v", err)
	}

	expected := []Kernel{{ID: 1}, {ID: 2}}
	if !reflect.DeepEqual(kernels, expected) {
		t.Errorf("Droplets.Kernels\n got=%#v\nwant=%#v", kernels, expected)
	}
}

func TestDroplets_Snapshots(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/droplets/12345/snapshots", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"snapshots": [{"id":1},{"id":2}]}`)
	})

	opt := &ListOptions{Page: 2}
	snapshots, _, err := client.Droplets.Snapshots(12345, opt)
	if err != nil {
		t.Errorf("Droplets.Snapshots returned error: %v", err)
	}

	expected := []Image{{ID: 1}, {ID: 2}}
	if !reflect.DeepEqual(snapshots, expected) {
		t.Errorf("Droplets.Snapshots\n got=%#v\nwant=%#v", snapshots, expected)
	}
}

func TestDroplets_Backups(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/droplets/12345/backups", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"backups": [{"id":1},{"id":2}]}`)
	})

	opt := &ListOptions{Page: 2}
	backups, _, err := client.Droplets.Backups(12345, opt)
	if err != nil {
		t.Errorf("Droplets.Backups returned error: %v", err)
	}

	expected := []Image{{ID: 1}, {ID: 2}}
	if !reflect.DeepEqual(backups, expected) {
		t.Errorf("Droplets.Backups\n got=%#v\nwant=%#v", backups, expected)
	}
}

func TestDroplets_Actions(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/droplets/12345/actions", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"actions": [{"id":1},{"id":2}]}`)
	})

	opt := &ListOptions{Page: 2}
	actions, _, err := client.Droplets.Actions(12345, opt)
	if err != nil {
		t.Errorf("Droplets.Actions returned error: %v", err)
	}

	expected := []Action{{ID: 1}, {ID: 2}}
	if !reflect.DeepEqual(actions, expected) {
		t.Errorf("Droplets.Actions\n got=%#v\nwant=%#v", actions, expected)
	}
}

func TestDroplets_Neighbors(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/droplets/12345/neighbors", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"droplets": [{"id":1},{"id":2}]}`)
	})

	neighbors, _, err := client.Droplets.Neighbors(12345)
	if err != nil {
		t.Errorf("Droplets.Neighbors returned error: %v", err)
	}

	expected := []Droplet{{ID: 1}, {ID: 2}}
	if !reflect.DeepEqual(neighbors, expected) {
		t.Errorf("Droplets.Neighbors\n got=%#v\nwant=%#v", neighbors, expected)
	}
}

func TestNetworkV4_String(t *testing.T) {
	network := &NetworkV4{
		IPAddress: "192.168.1.2",
		Netmask:   "255.255.255.0",
		Gateway:   "192.168.1.1",
	}

	stringified := network.String()
	expected := `godo.NetworkV4{IPAddress:"192.168.1.2", Netmask:"255.255.255.0", Gateway:"192.168.1.1", Type:""}`
	if expected != stringified {
		t.Errorf("NetworkV4.String\n got=%#v\nwant=%#v", stringified, expected)
	}

}

func TestNetworkV6_String(t *testing.T) {
	network := &NetworkV6{
		IPAddress: "2604:A880:0800:0010:0000:0000:02DD:4001",
		Netmask:   64,
		Gateway:   "2604:A880:0800:0010:0000:0000:0000:0001",
	}
	stringified := network.String()
	expected := `godo.NetworkV6{IPAddress:"2604:A880:0800:0010:0000:0000:02DD:4001", Netmask:64, Gateway:"2604:A880:0800:0010:0000:0000:0000:0001", Type:""}`
	if expected != stringified {
		t.Errorf("NetworkV6.String\n got=%#v\nwant=%#v", stringified, expected)
	}
}

func TestDroplets_IPMethods(t *testing.T) {
	var d Droplet

	ipv6 := "1000:1000:1000:1000:0000:0000:004D:B001"

	d.Networks = &Networks{
		V4: []NetworkV4{
			{IPAddress: "192.168.0.1", Type: "public"},
			{IPAddress: "10.0.0.1", Type: "private"},
		},
		V6: []NetworkV6{
			{IPAddress: ipv6, Type: "public"},
		},
	}

	ip, err := d.PublicIPv4()
	if err != nil {
		t.Errorf("unknown error")
	}

	if got, expected := ip, "192.168.0.1"; got != expected {
		t.Errorf("Droplet.PublicIPv4 returned %s; expected %s", got, expected)
	}

	ip, err = d.PrivateIPv4()
	if err != nil {
		t.Errorf("unknown error")
	}

	if got, expected := ip, "10.0.0.1"; got != expected {
		t.Errorf("Droplet.PrivateIPv4 returned %s; expected %s", got, expected)
	}

	ip, err = d.PublicIPv6()
	if err != nil {
		t.Errorf("unknown error")
	}

	if got, expected := ip, ipv6; got != expected {
		t.Errorf("Droplet.PublicIPv6 returned %s; expected %s", got, expected)
	}
}
