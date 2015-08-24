// Copyright 2015 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package docker

import (
	"encoding/json"
	"net/http"
	"net/url"
	"reflect"
	"testing"
)

func TestListNetworks(t *testing.T) {
	jsonNetworks := `[
     {
             "ID": "8dfafdbc3a40",
             "Name": "blah",
             "Type": "bridge",
             "Endpoints":[{"ID": "918c11c8288a", "Name": "dsafdsaf", "Network": "8dfafdbc3a40"}]
     },
     {
             "ID": "9fb1e39c",
             "Name": "foo",
             "Type": "bridge",
             "Endpoints":[{"ID": "c080be979dda", "Name": "lllll2222", "Network": "9fb1e39c"}]
     }
]`
	var expected []Network
	err := json.Unmarshal([]byte(jsonNetworks), &expected)
	if err != nil {
		t.Fatal(err)
	}
	client := newTestClient(&FakeRoundTripper{message: jsonNetworks, status: http.StatusOK})
	containers, err := client.ListNetworks()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(containers, expected) {
		t.Errorf("ListNetworks: Expected %#v. Got %#v.", expected, containers)
	}
}

func TestNetworkInfo(t *testing.T) {
	jsonNetwork := `{
             "ID": "8dfafdbc3a40",
             "Name": "blah",
             "Type": "bridge",
             "Endpoints":[{"ID": "918c11c8288a", "Name": "dsafdsaf", "Network": "8dfafdbc3a40"}]
        }`
	var expected Network
	err := json.Unmarshal([]byte(jsonNetwork), &expected)
	if err != nil {
		t.Fatal(err)
	}
	fakeRT := &FakeRoundTripper{message: jsonNetwork, status: http.StatusOK}
	client := newTestClient(fakeRT)
	id := "8dfafdbc3a40"
	network, err := client.NetworkInfo(id)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(*network, expected) {
		t.Errorf("NetworkInfo(%q): Expected %#v. Got %#v.", id, expected, network)
	}
	expectedURL, _ := url.Parse(client.getURL("/networks/8dfafdbc3a40"))
	if gotPath := fakeRT.requests[0].URL.Path; gotPath != expectedURL.Path {
		t.Errorf("NetworkInfo(%q): Wrong path in request. Want %q. Got %q.", id, expectedURL.Path, gotPath)
	}
}

func TestNetworkCreate(t *testing.T) {
	jsonID := `{"ID": "8dfafdbc3a40"}`
	jsonNetwork := `{
             "ID": "8dfafdbc3a40",
             "Name": "foobar",
             "Type": "bridge"
        }`
	var expected Network
	err := json.Unmarshal([]byte(jsonNetwork), &expected)
	if err != nil {
		t.Fatal(err)
	}

	client := newTestClient(&FakeRoundTripper{message: jsonID, status: http.StatusOK})
	opts := CreateNetworkOptions{"foobar", "bridge", nil}
	network, err := client.CreateNetwork(opts)
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(*network, expected) {
		t.Errorf("CreateNetwork: Expected %#v. Got %#v.", expected, network)
	}
}
