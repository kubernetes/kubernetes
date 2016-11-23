package godo

import (
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"testing"
)

func TestFloatingIPsActions_Assign(t *testing.T) {
	setup()
	defer teardown()
	dropletID := 12345
	assignRequest := &ActionRequest{
		"droplet_id": float64(dropletID), // encoding/json decodes numbers as floats
		"type":       "assign",
	}

	mux.HandleFunc("/v2/floating_ips/192.168.0.1/actions", func(w http.ResponseWriter, r *http.Request) {
		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, assignRequest) {
			t.Errorf("Request body = %#v, expected %#v", v, assignRequest)
		}

		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)

	})

	assign, _, err := client.FloatingIPActions.Assign("192.168.0.1", 12345)
	if err != nil {
		t.Errorf("FloatingIPsActions.Assign returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(assign, expected) {
		t.Errorf("FloatingIPsActions.Assign returned %+v, expected %+v", assign, expected)
	}
}

func TestFloatingIPsActions_Unassign(t *testing.T) {
	setup()
	defer teardown()

	unassignRequest := &ActionRequest{
		"type": "unassign",
	}

	mux.HandleFunc("/v2/floating_ips/192.168.0.1/actions", func(w http.ResponseWriter, r *http.Request) {
		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, unassignRequest) {
			t.Errorf("Request body = %+v, expected %+v", v, unassignRequest)
		}

		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)
	})

	action, _, err := client.FloatingIPActions.Unassign("192.168.0.1")
	if err != nil {
		t.Errorf("FloatingIPsActions.Get returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("FloatingIPsActions.Get returned %+v, expected %+v", action, expected)
	}
}

func TestFloatingIPsActions_Get(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/floating_ips/192.168.0.1/actions/456", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)
	})

	action, _, err := client.FloatingIPActions.Get("192.168.0.1", 456)
	if err != nil {
		t.Errorf("FloatingIPsActions.Get returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("FloatingIPsActions.Get returned %+v, expected %+v", action, expected)
	}
}

func TestFloatingIPsActions_List(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/floating_ips/192.168.0.1/actions", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprintf(w, `{"actions":[{"status":"in-progress"}]}`)
	})

	actions, _, err := client.FloatingIPActions.List("192.168.0.1", nil)
	if err != nil {
		t.Errorf("FloatingIPsActions.List returned error: %v", err)
	}

	expected := []Action{{Status: "in-progress"}}
	if !reflect.DeepEqual(actions, expected) {
		t.Errorf("FloatingIPsActions.List returned %+v, expected %+v", actions, expected)
	}
}

func TestFloatingIPsActions_ListMultiplePages(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/floating_ips/192.168.0.1/actions", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"actions":[{"status":"in-progress"}], "links":{"pages":{"next":"http://example.com/v2/floating_ips/192.168.0.1/actions?page=2"}}}`)
	})

	_, resp, err := client.FloatingIPActions.List("192.168.0.1", nil)
	if err != nil {
		t.Errorf("FloatingIPsActions.List returned error: %v", err)
	}

	checkCurrentPage(t, resp, 1)
}

func TestFloatingIPsActions_ListPageByNumber(t *testing.T) {
	setup()
	defer teardown()

	jBlob := `
	{
		"actions":[{"status":"in-progress"}],
		"links":{
			"pages":{
				"next":"http://example.com/v2/regions/?page=3",
				"prev":"http://example.com/v2/regions/?page=1",
				"last":"http://example.com/v2/regions/?page=3",
				"first":"http://example.com/v2/regions/?page=1"
			}
		}
	}`

	mux.HandleFunc("/v2/floating_ips/192.168.0.1/actions", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, jBlob)
	})

	opt := &ListOptions{Page: 2}
	_, resp, err := client.FloatingIPActions.List("192.168.0.1", opt)
	if err != nil {
		t.Errorf("FloatingIPsActions.List returned error: %v", err)
	}

	checkCurrentPage(t, resp, 2)
}
