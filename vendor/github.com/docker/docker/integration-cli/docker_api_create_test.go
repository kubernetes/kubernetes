package main

import (
	"fmt"
	"net/http"
	"time"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/request"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestAPICreateWithNotExistImage(c *check.C) {
	name := "test"
	config := map[string]interface{}{
		"Image":   "test456:v1",
		"Volumes": map[string]struct{}{"/tmp": {}},
	}

	status, body, err := request.SockRequest("POST", "/containers/create?name="+name, config, daemonHost())
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusNotFound)
	expected := "No such image: test456:v1"
	c.Assert(getErrorMessage(c, body), checker.Contains, expected)

	config2 := map[string]interface{}{
		"Image":   "test456",
		"Volumes": map[string]struct{}{"/tmp": {}},
	}

	status, body, err = request.SockRequest("POST", "/containers/create?name="+name, config2, daemonHost())
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusNotFound)
	expected = "No such image: test456:latest"
	c.Assert(getErrorMessage(c, body), checker.Equals, expected)

	config3 := map[string]interface{}{
		"Image": "sha256:0cb40641836c461bc97c793971d84d758371ed682042457523e4ae701efeaaaa",
	}

	status, body, err = request.SockRequest("POST", "/containers/create?name="+name, config3, daemonHost())
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusNotFound)
	expected = "No such image: sha256:0cb40641836c461bc97c793971d84d758371ed682042457523e4ae701efeaaaa"
	c.Assert(getErrorMessage(c, body), checker.Equals, expected)

}

// Test for #25099
func (s *DockerSuite) TestAPICreateEmptyEnv(c *check.C) {
	name := "test1"
	config := map[string]interface{}{
		"Image": "busybox",
		"Env":   []string{"", "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"},
		"Cmd":   []string{"true"},
	}

	status, body, err := request.SockRequest("POST", "/containers/create?name="+name, config, daemonHost())
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusInternalServerError)
	expected := "invalid environment variable:"
	c.Assert(getErrorMessage(c, body), checker.Contains, expected)

	name = "test2"
	config = map[string]interface{}{
		"Image": "busybox",
		"Env":   []string{"=", "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"},
		"Cmd":   []string{"true"},
	}
	status, body, err = request.SockRequest("POST", "/containers/create?name="+name, config, daemonHost())
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusInternalServerError)
	expected = "invalid environment variable: ="
	c.Assert(getErrorMessage(c, body), checker.Contains, expected)

	name = "test3"
	config = map[string]interface{}{
		"Image": "busybox",
		"Env":   []string{"=foo", "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"},
		"Cmd":   []string{"true"},
	}
	status, body, err = request.SockRequest("POST", "/containers/create?name="+name, config, daemonHost())
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusInternalServerError)
	expected = "invalid environment variable: =foo"
	c.Assert(getErrorMessage(c, body), checker.Contains, expected)
}

func (s *DockerSuite) TestAPICreateWithInvalidHealthcheckParams(c *check.C) {
	// test invalid Interval in Healthcheck: less than 0s
	name := "test1"
	config := map[string]interface{}{
		"Image": "busybox",
		"Healthcheck": map[string]interface{}{
			"Interval": -10 * time.Millisecond,
			"Timeout":  time.Second,
			"Retries":  int(1000),
		},
	}

	status, body, err := request.SockRequest("POST", "/containers/create?name="+name, config, daemonHost())
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusInternalServerError)
	expected := fmt.Sprintf("Interval in Healthcheck cannot be less than %s", container.MinimumDuration)
	c.Assert(getErrorMessage(c, body), checker.Contains, expected)

	// test invalid Interval in Healthcheck: larger than 0s but less than 1ms
	name = "test2"
	config = map[string]interface{}{
		"Image": "busybox",
		"Healthcheck": map[string]interface{}{
			"Interval": 500 * time.Microsecond,
			"Timeout":  time.Second,
			"Retries":  int(1000),
		},
	}
	status, body, err = request.SockRequest("POST", "/containers/create?name="+name, config, daemonHost())
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusInternalServerError)
	c.Assert(getErrorMessage(c, body), checker.Contains, expected)

	// test invalid Timeout in Healthcheck: less than 1ms
	name = "test3"
	config = map[string]interface{}{
		"Image": "busybox",
		"Healthcheck": map[string]interface{}{
			"Interval": time.Second,
			"Timeout":  -100 * time.Millisecond,
			"Retries":  int(1000),
		},
	}
	status, body, err = request.SockRequest("POST", "/containers/create?name="+name, config, daemonHost())
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusInternalServerError)
	expected = fmt.Sprintf("Timeout in Healthcheck cannot be less than %s", container.MinimumDuration)
	c.Assert(getErrorMessage(c, body), checker.Contains, expected)

	// test invalid Retries in Healthcheck: less than 0
	name = "test4"
	config = map[string]interface{}{
		"Image": "busybox",
		"Healthcheck": map[string]interface{}{
			"Interval": time.Second,
			"Timeout":  time.Second,
			"Retries":  int(-10),
		},
	}
	status, body, err = request.SockRequest("POST", "/containers/create?name="+name, config, daemonHost())
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusInternalServerError)
	expected = "Retries in Healthcheck cannot be negative"
	c.Assert(getErrorMessage(c, body), checker.Contains, expected)

	// test invalid StartPeriod in Healthcheck: not 0 and less than 1ms
	name = "test3"
	config = map[string]interface{}{
		"Image": "busybox",
		"Healthcheck": map[string]interface{}{
			"Interval":    time.Second,
			"Timeout":     time.Second,
			"Retries":     int(1000),
			"StartPeriod": 100 * time.Microsecond,
		},
	}
	status, body, err = request.SockRequest("POST", "/containers/create?name="+name, config, daemonHost())
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusInternalServerError)
	expected = fmt.Sprintf("StartPeriod in Healthcheck cannot be less than %s", container.MinimumDuration)
	c.Assert(getErrorMessage(c, body), checker.Contains, expected)
}
