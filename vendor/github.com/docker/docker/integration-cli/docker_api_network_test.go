// +build experimental

package main

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/go-check/check"
)

func isNetworkAvailable(c *check.C, name string) bool {
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

func (s *DockerSuite) TestNetworkApiGetAll(c *check.C) {
	defaults := []string{"bridge", "host", "none"}
	for _, nn := range defaults {
		if !isNetworkAvailable(c, nn) {
			c.Fatalf("Missing Default network : %s", nn)
		}
	}
}

func (s *DockerSuite) TestNetworkApiCreateDelete(c *check.C) {
	name := "testnetwork"
	config := map[string]interface{}{
		"name":         name,
		"network_type": "bridge",
	}

	status, resp, err := sockRequest("POST", "/networks", config)
	c.Assert(status, check.Equals, http.StatusCreated)
	c.Assert(err, check.IsNil)

	if !isNetworkAvailable(c, name) {
		c.Fatalf("Network %s not found", name)
	}

	var id string
	err = json.Unmarshal(resp, &id)
	if err != nil {
		c.Fatal(err)
	}

	status, _, err = sockRequest("DELETE", fmt.Sprintf("/networks/%s", id), nil)
	c.Assert(status, check.Equals, http.StatusOK)
	c.Assert(err, check.IsNil)

	if isNetworkAvailable(c, name) {
		c.Fatalf("Network %s not deleted", name)
	}
}
