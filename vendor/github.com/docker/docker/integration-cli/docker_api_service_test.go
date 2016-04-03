// +build experimental

package main

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/go-check/check"
)

func isServiceAvailable(c *check.C, name string, network string) bool {
	status, body, err := sockRequest("GET", "/services", nil)
	c.Assert(status, check.Equals, http.StatusOK)
	c.Assert(err, check.IsNil)

	var inspectJSON []struct {
		Name    string
		ID      string
		Network string
	}
	if err = json.Unmarshal(body, &inspectJSON); err != nil {
		c.Fatalf("unable to unmarshal response body: %v", err)
	}
	for _, s := range inspectJSON {
		if s.Name == name && s.Network == network {
			return true
		}
	}
	return false

}

func isServiceNetworkAvailable(c *check.C, name string) bool {
	status, body, err := sockRequest("GET", "/networks", nil)
	c.Assert(status, check.Equals, http.StatusOK)
	c.Assert(err, check.IsNil)

	var inspectJSON []struct {
		Name string
		ID   string
		Type string
	}
	if err = json.Unmarshal(body, &inspectJSON); err != nil {
		c.Fatalf("unable to unmarshal response body: %v", err)
	}
	for _, n := range inspectJSON {
		if n.Name == name {
			return true
		}
	}
	return false

}

func (s *DockerSuite) TestServiceApiCreateDelete(c *check.C) {
	name := "testnetwork"
	config := map[string]interface{}{
		"name":         name,
		"network_type": "bridge",
	}

	status, resp, err := sockRequest("POST", "/networks", config)
	c.Assert(status, check.Equals, http.StatusCreated)
	c.Assert(err, check.IsNil)

	if !isServiceNetworkAvailable(c, name) {
		c.Fatalf("Network %s not found", name)
	}

	var nid string
	err = json.Unmarshal(resp, &nid)
	if err != nil {
		c.Fatal(err)
	}

	sname := "service1"
	sconfig := map[string]interface{}{
		"name":         sname,
		"network_name": name,
	}

	status, resp, err = sockRequest("POST", "/services", sconfig)
	c.Assert(status, check.Equals, http.StatusCreated)
	c.Assert(err, check.IsNil)

	if !isServiceAvailable(c, sname, name) {
		c.Fatalf("Service %s.%s not found", sname, name)
	}

	var id string
	err = json.Unmarshal(resp, &id)
	if err != nil {
		c.Fatal(err)
	}

	status, _, err = sockRequest("DELETE", fmt.Sprintf("/services/%s", id), nil)
	c.Assert(status, check.Equals, http.StatusOK)
	c.Assert(err, check.IsNil)

	if isServiceAvailable(c, sname, name) {
		c.Fatalf("Service %s.%s not deleted", sname, name)
	}

	status, _, err = sockRequest("DELETE", fmt.Sprintf("/networks/%s", nid), nil)
	c.Assert(status, check.Equals, http.StatusOK)
	c.Assert(err, check.IsNil)

	if isNetworkAvailable(c, name) {
		c.Fatalf("Network %s not deleted", name)
	}
}
