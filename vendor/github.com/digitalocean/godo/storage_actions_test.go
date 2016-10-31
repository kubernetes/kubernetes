package godo

import (
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"testing"
)

func TestStoragesActions_Attach(t *testing.T) {
	setup()
	defer teardown()
	const (
		volumeID  = "98d414c6-295e-4e3a-ac58-eb9456c1e1d1"
		dropletID = 12345
	)

	attachRequest := &ActionRequest{
		"type":       "attach",
		"droplet_id": float64(dropletID), // encoding/json decodes numbers as floats
	}

	mux.HandleFunc("/v2/volumes/"+volumeID+"/actions", func(w http.ResponseWriter, r *http.Request) {
		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, attachRequest) {
			t.Errorf("want=%#v", attachRequest)
			t.Errorf("got=%#v", v)
		}
		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)
	})

	_, _, err := client.StorageActions.Attach(volumeID, dropletID)
	if err != nil {
		t.Errorf("StoragesActions.Attach returned error: %v", err)
	}
}

func TestStoragesActions_Detach(t *testing.T) {
	setup()
	defer teardown()
	volumeID := "98d414c6-295e-4e3a-ac58-eb9456c1e1d1"

	detachRequest := &ActionRequest{
		"type": "detach",
	}

	mux.HandleFunc("/v2/volumes/"+volumeID+"/actions", func(w http.ResponseWriter, r *http.Request) {
		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, detachRequest) {
			t.Errorf("want=%#v", detachRequest)
			t.Errorf("got=%#v", v)
		}
		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)
	})

	_, _, err := client.StorageActions.Detach(volumeID)
	if err != nil {
		t.Errorf("StoragesActions.Detach returned error: %v", err)
	}
}

func TestStoragesActions_DetachByDropletID(t *testing.T) {
	setup()
	defer teardown()
	volumeID := "98d414c6-295e-4e3a-ac58-eb9456c1e1d1"
	dropletID := 123456

	detachByDropletIDRequest := &ActionRequest{
		"type":       "detach",
		"droplet_id": float64(dropletID), // encoding/json decodes numbers as floats
	}

	mux.HandleFunc("/v2/volumes/"+volumeID+"/actions", func(w http.ResponseWriter, r *http.Request) {
		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, detachByDropletIDRequest) {
			t.Errorf("want=%#v", detachByDropletIDRequest)
			t.Errorf("got=%#v", v)
		}
		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)
	})

	_, _, err := client.StorageActions.DetachByDropletID(volumeID, dropletID)
	if err != nil {
		t.Errorf("StoragesActions.DetachByDropletID returned error: %v", err)
	}
}

func TestStorageActions_Get(t *testing.T) {
	setup()
	defer teardown()
	volumeID := "98d414c6-295e-4e3a-ac58-eb9456c1e1d1"

	mux.HandleFunc("/v2/volumes/"+volumeID+"/actions/456", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)
	})

	action, _, err := client.StorageActions.Get(volumeID, 456)
	if err != nil {
		t.Errorf("StorageActions.Get returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("StorageActions.Get returned %+v, expected %+v", action, expected)
	}
}

func TestStorageActions_List(t *testing.T) {
	setup()
	defer teardown()
	volumeID := "98d414c6-295e-4e3a-ac58-eb9456c1e1d1"

	mux.HandleFunc("/v2/volumes/"+volumeID+"/actions", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprintf(w, `{"actions":[{"status":"in-progress"}]}`)
	})

	actions, _, err := client.StorageActions.List(volumeID, nil)
	if err != nil {
		t.Errorf("StorageActions.List returned error: %v", err)
	}

	expected := []Action{Action{Status: "in-progress"}}
	if !reflect.DeepEqual(actions, expected) {
		t.Errorf("StorageActions.List returned %+v, expected %+v", actions, expected)
	}
}

func TestStoragesActions_Resize(t *testing.T) {
	setup()
	defer teardown()
	volumeID := "98d414c6-295e-4e3a-ac58-eb9456c1e1d1"

	resizeRequest := &ActionRequest{
		"type":           "resize",
		"size_gigabytes": float64(500),
		"region":         "nyc1",
	}

	mux.HandleFunc("/v2/volumes/"+volumeID+"/actions", func(w http.ResponseWriter, r *http.Request) {
		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, resizeRequest) {
			t.Errorf("want=%#v", resizeRequest)
			t.Errorf("got=%#v", v)
		}
		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)
	})

	_, _, err := client.StorageActions.Resize(volumeID, 500, "nyc1")
	if err != nil {
		t.Errorf("StoragesActions.Resize returned error: %v", err)
	}
}
