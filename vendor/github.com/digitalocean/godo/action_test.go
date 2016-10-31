package godo

import (
	"fmt"
	"net/http"
	"reflect"
	"testing"
	"time"
)

func TestAction_List(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/actions", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"actions": [{"id":1},{"id":2}]}`)
		testMethod(t, r, "GET")
	})

	actions, _, err := client.Actions.List(nil)
	if err != nil {
		t.Fatalf("unexpected error: %s", err)
	}

	expected := []Action{{ID: 1}, {ID: 2}}
	if len(actions) != len(expected) || actions[0].ID != expected[0].ID || actions[1].ID != expected[1].ID {
		t.Fatalf("unexpected response")
	}
}

func TestAction_ListActionMultiplePages(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/actions", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"actions": [{"id":1},{"id":2}], "links":{"pages":{"next":"http://example.com/v2/droplets/?page=2"}}}`)
		testMethod(t, r, "GET")
	})

	_, resp, err := client.Actions.List(nil)
	if err != nil {
		t.Fatal(nil)
	}

	checkCurrentPage(t, resp, 1)
}

func TestAction_RetrievePageByNumber(t *testing.T) {
	setup()
	defer teardown()

	jBlob := `
	{
		"actions": [{"id":1},{"id":2}],
		"links":{
			"pages":{
				"next":"http://example.com/v2/actions/?page=3",
				"prev":"http://example.com/v2/actions/?page=1",
				"last":"http://example.com/v2/actions/?page=3",
				"first":"http://example.com/v2/actions/?page=1"
			}
		}
	}`

	mux.HandleFunc("/v2/actions", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, jBlob)
	})

	opt := &ListOptions{Page: 2}
	_, resp, err := client.Actions.List(opt)
	if err != nil {
		t.Fatal(err)
	}

	checkCurrentPage(t, resp, 2)
}

func TestAction_Get(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/actions/12345", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"action": {"id":12345,"region":{"name":"name","slug":"slug","available":true,"sizes":["512mb"],"features":["virtio"]},"region_slug":"slug"}}`)
		testMethod(t, r, "GET")
	})

	action, _, err := client.Actions.Get(12345)
	if err != nil {
		t.Fatalf("unexpected error: %s", err)
	}

	if action.ID != 12345 {
		t.Fatalf("unexpected response")
	}

	region := &Region{
		Name:      "name",
		Slug:      "slug",
		Available: true,
		Sizes:     []string{"512mb"},
		Features:  []string{"virtio"},
	}
	if !reflect.DeepEqual(action.Region, region) {
		t.Fatalf("unexpected response, invalid region")
	}

	if action.RegionSlug != "slug" {
		t.Fatalf("unexpected response, invalid region slug")
	}
}

func TestAction_String(t *testing.T) {
	pt, err := time.Parse(time.RFC3339, "2014-05-08T20:36:47Z")
	if err != nil {
		t.Fatalf("unexpected error: %s", err)
	}

	startedAt := &Timestamp{
		Time: pt,
	}
	action := &Action{
		ID:        1,
		Status:    "in-progress",
		Type:      "transfer",
		StartedAt: startedAt,
	}

	stringified := action.String()
	expected := `godo.Action{ID:1, Status:"in-progress", Type:"transfer", ` +
		`StartedAt:godo.Timestamp{2014-05-08 20:36:47 +0000 UTC}, ` +
		`ResourceID:0, ResourceType:"", RegionSlug:""}`
	if expected != stringified {
		t.Errorf("Action.Stringify returned %+v, expected %+v", stringified, expected)
	}
}
