package godo

import (
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"testing"
)

func TestFloatingIPs_ListFloatingIPs(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/floating_ips", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"floating_ips": [{"region":{"slug":"nyc3"},"droplet":{"id":1},"ip":"192.168.0.1"},{"region":{"slug":"nyc3"},"droplet":{"id":2},"ip":"192.168.0.2"}]}`)
	})

	floatingIPs, _, err := client.FloatingIPs.List(nil)
	if err != nil {
		t.Errorf("FloatingIPs.List returned error: %v", err)
	}

	expected := []FloatingIP{
		{Region: &Region{Slug: "nyc3"}, Droplet: &Droplet{ID: 1}, IP: "192.168.0.1"},
		{Region: &Region{Slug: "nyc3"}, Droplet: &Droplet{ID: 2}, IP: "192.168.0.2"},
	}
	if !reflect.DeepEqual(floatingIPs, expected) {
		t.Errorf("FloatingIPs.List returned %+v, expected %+v", floatingIPs, expected)
	}
}

func TestFloatingIPs_ListFloatingIPsMultiplePages(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/floating_ips", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"floating_ips": [{"region":{"slug":"nyc3"},"droplet":{"id":1},"ip":"192.168.0.1"},{"region":{"slug":"nyc3"},"droplet":{"id":2},"ip":"192.168.0.2"}], "links":{"pages":{"next":"http://example.com/v2/floating_ips/?page=2"}}}`)
	})

	_, resp, err := client.FloatingIPs.List(nil)
	if err != nil {
		t.Fatal(err)
	}

	checkCurrentPage(t, resp, 1)
}

func TestFloatingIPs_RetrievePageByNumber(t *testing.T) {
	setup()
	defer teardown()

	jBlob := `
	{
		"floating_ips": [{"region":{"slug":"nyc3"},"droplet":{"id":1},"ip":"192.168.0.1"},{"region":{"slug":"nyc3"},"droplet":{"id":2},"ip":"192.168.0.2"}],
		"links":{
			"pages":{
				"next":"http://example.com/v2/floating_ips/?page=3",
				"prev":"http://example.com/v2/floating_ips/?page=1",
				"last":"http://example.com/v2/floating_ips/?page=3",
				"first":"http://example.com/v2/floating_ips/?page=1"
			}
		}
	}`

	mux.HandleFunc("/v2/floating_ips", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, jBlob)
	})

	opt := &ListOptions{Page: 2}
	_, resp, err := client.FloatingIPs.List(opt)
	if err != nil {
		t.Fatal(err)
	}

	checkCurrentPage(t, resp, 2)
}

func TestFloatingIPs_Get(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/floating_ips/192.168.0.1", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"floating_ip":{"region":{"slug":"nyc3"},"droplet":{"id":1},"ip":"192.168.0.1"}}`)
	})

	floatingIP, _, err := client.FloatingIPs.Get("192.168.0.1")
	if err != nil {
		t.Errorf("domain.Get returned error: %v", err)
	}

	expected := &FloatingIP{Region: &Region{Slug: "nyc3"}, Droplet: &Droplet{ID: 1}, IP: "192.168.0.1"}
	if !reflect.DeepEqual(floatingIP, expected) {
		t.Errorf("FloatingIPs.Get returned %+v, expected %+v", floatingIP, expected)
	}
}

func TestFloatingIPs_Create(t *testing.T) {
	setup()
	defer teardown()

	createRequest := &FloatingIPCreateRequest{
		Region:    "nyc3",
		DropletID: 1,
	}

	mux.HandleFunc("/v2/floating_ips", func(w http.ResponseWriter, r *http.Request) {
		v := new(FloatingIPCreateRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatal(err)
		}

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, createRequest) {
			t.Errorf("Request body = %+v, expected %+v", v, createRequest)
		}

		fmt.Fprint(w, `{"floating_ip":{"region":{"slug":"nyc3"},"droplet":{"id":1},"ip":"192.168.0.1"}}`)
	})

	floatingIP, _, err := client.FloatingIPs.Create(createRequest)
	if err != nil {
		t.Errorf("FloatingIPs.Create returned error: %v", err)
	}

	expected := &FloatingIP{Region: &Region{Slug: "nyc3"}, Droplet: &Droplet{ID: 1}, IP: "192.168.0.1"}
	if !reflect.DeepEqual(floatingIP, expected) {
		t.Errorf("FloatingIPs.Create returned %+v, expected %+v", floatingIP, expected)
	}
}

func TestFloatingIPs_Destroy(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/floating_ips/192.168.0.1", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "DELETE")
	})

	_, err := client.FloatingIPs.Delete("192.168.0.1")
	if err != nil {
		t.Errorf("FloatingIPs.Delete returned error: %v", err)
	}
}
