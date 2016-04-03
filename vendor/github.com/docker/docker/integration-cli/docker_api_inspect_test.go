package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/go-check/check"
)

func (s *DockerSuite) TestInspectApiContainerResponse(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "true")

	cleanedContainerID := strings.TrimSpace(out)
	keysBase := []string{"Id", "State", "Created", "Path", "Args", "Config", "Image", "NetworkSettings",
		"ResolvConfPath", "HostnamePath", "HostsPath", "LogPath", "Name", "Driver", "ExecDriver", "MountLabel", "ProcessLabel", "GraphDriver"}

	cases := []struct {
		version string
		keys    []string
	}{
		{"1.20", append(keysBase, "Mounts")},
		{"1.19", append(keysBase, "Volumes", "VolumesRW")},
	}

	for _, cs := range cases {
		endpoint := fmt.Sprintf("/v%s/containers/%s/json", cs.version, cleanedContainerID)

		status, body, err := sockRequest("GET", endpoint, nil)
		c.Assert(status, check.Equals, http.StatusOK)
		c.Assert(err, check.IsNil)

		var inspectJSON map[string]interface{}
		if err = json.Unmarshal(body, &inspectJSON); err != nil {
			c.Fatalf("unable to unmarshal body for version %s: %v", cs.version, err)
		}

		for _, key := range cs.keys {
			if _, ok := inspectJSON[key]; !ok {
				c.Fatalf("%s does not exist in response for version %s", key, cs.version)
			}
		}

		//Issue #6830: type not properly converted to JSON/back
		if _, ok := inspectJSON["Path"].(bool); ok {
			c.Fatalf("Path of `true` should not be converted to boolean `true` via JSON marshalling")
		}
	}
}
