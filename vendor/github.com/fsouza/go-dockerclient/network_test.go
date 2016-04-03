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
             "Driver": "bridge"
        }`
	var expected Network
	err := json.Unmarshal([]byte(jsonNetwork), &expected)
	if err != nil {
		t.Fatal(err)
	}

	client := newTestClient(&FakeRoundTripper{message: jsonID, status: http.StatusOK})
	opts := CreateNetworkOptions{"foobar", false, "bridge", IPAMOptions{}, nil}
	network, err := client.CreateNetwork(opts)
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(*network, expected) {
		t.Errorf("CreateNetwork: Expected %#v. Got %#v.", expected, network)
	}
}

func TestNetworkRemove(t *testing.T) {
	id := "8dfafdbc3a40"
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusNoContent}
	client := newTestClient(fakeRT)
	err := client.RemoveNetwork(id)
	if err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	expectedMethod := "DELETE"
	if req.Method != expectedMethod {
		t.Errorf("RemoveNetwork(%q): Wrong HTTP method. Want %s. Got %s.", id, expectedMethod, req.Method)
	}
	u, _ := url.Parse(client.getURL("/networks/" + id))
	if req.URL.Path != u.Path {
		t.Errorf("RemoveNetwork(%q): Wrong request path. Want %q. Got %q.", id, u.Path, req.URL.Path)
	}
}

func TestNetworkConnect(t *testing.T) {
	id := "8dfafdbc3a40"
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusNoContent}
	client := newTestClient(fakeRT)
	opts := NetworkConnectionOptions{"foobar"}
	err := client.ConnectNetwork(id, opts)
	if err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	expectedMethod := "POST"
	if req.Method != expectedMethod {
		t.Errorf("ConnectNetwork(%q): Wrong HTTP method. Want %s. Got %s.", id, expectedMethod, req.Method)
	}
	u, _ := url.Parse(client.getURL("/networks/" + id + "/connect"))
	if req.URL.Path != u.Path {
		t.Errorf("ConnectNetwork(%q): Wrong request path. Want %q. Got %q.", id, u.Path, req.URL.Path)
	}
}

func TestNetworkConnectNotFound(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "no such network container", status: http.StatusNotFound})
	opts := NetworkConnectionOptions{"foobar"}
	err := client.ConnectNetwork("8dfafdbc3a40", opts)
	if serr, ok := err.(*NoSuchNetworkOrContainer); !ok {
		t.Errorf("ConnectNetwork: wrong error type: %s.", serr)
	}
}

func TestNetworkDisconnect(t *testing.T) {
	id := "8dfafdbc3a40"
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusNoContent}
	client := newTestClient(fakeRT)
	opts := NetworkConnectionOptions{"foobar"}
	err := client.DisconnectNetwork(id, opts)
	if err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	expectedMethod := "POST"
	if req.Method != expectedMethod {
		t.Errorf("DisconnectNetwork(%q): Wrong HTTP method. Want %s. Got %s.", id, expectedMethod, req.Method)
	}
	u, _ := url.Parse(client.getURL("/networks/" + id + "/disconnect"))
	if req.URL.Path != u.Path {
		t.Errorf("DisconnectNetwork(%q): Wrong request path. Want %q. Got %q.", id, u.Path, req.URL.Path)
	}
}

func TestNetworkDisconnectNotFound(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "no such network container", status: http.StatusNotFound})
	opts := NetworkConnectionOptions{"foobar"}
	err := client.DisconnectNetwork("8dfafdbc3a40", opts)
	if serr, ok := err.(*NoSuchNetworkOrContainer); !ok {
		t.Errorf("DisconnectNetwork: wrong error type: %s.", serr)
	}
}
