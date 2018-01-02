// This file will be removed when we completely drop support for
// passing HostConfig to container start API.

package main

import (
	"net/http"
	"strings"

	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/request"
	"github.com/docker/docker/pkg/testutil"
	"github.com/go-check/check"
)

func formatV123StartAPIURL(url string) string {
	return "/v1.23" + url
}

func (s *DockerSuite) TestDeprecatedContainerAPIStartHostConfig(c *check.C) {
	name := "test-deprecated-api-124"
	dockerCmd(c, "create", "--name", name, "busybox")
	config := map[string]interface{}{
		"Binds": []string{"/aa:/bb"},
	}
	status, body, err := request.SockRequest("POST", "/containers/"+name+"/start", config, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusBadRequest)
	c.Assert(string(body), checker.Contains, "was deprecated since v1.10")
}

func (s *DockerSuite) TestDeprecatedContainerAPIStartVolumeBinds(c *check.C) {
	// TODO Windows CI: Investigate further why this fails on Windows to Windows CI.
	testRequires(c, DaemonIsLinux)
	path := "/foo"
	if testEnv.DaemonPlatform() == "windows" {
		path = `c:\foo`
	}
	name := "testing"
	config := map[string]interface{}{
		"Image":   "busybox",
		"Volumes": map[string]struct{}{path: {}},
	}

	status, _, err := request.SockRequest("POST", formatV123StartAPIURL("/containers/create?name="+name), config, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusCreated)

	bindPath := testutil.RandomTmpDirPath("test", testEnv.DaemonPlatform())
	config = map[string]interface{}{
		"Binds": []string{bindPath + ":" + path},
	}
	status, _, err = request.SockRequest("POST", formatV123StartAPIURL("/containers/"+name+"/start"), config, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusNoContent)

	pth, err := inspectMountSourceField(name, path)
	c.Assert(err, checker.IsNil)
	c.Assert(pth, checker.Equals, bindPath, check.Commentf("expected volume host path to be %s, got %s", bindPath, pth))
}

// Test for GH#10618
func (s *DockerSuite) TestDeprecatedContainerAPIStartDupVolumeBinds(c *check.C) {
	// TODO Windows to Windows CI - Port this
	testRequires(c, DaemonIsLinux)
	name := "testdups"
	config := map[string]interface{}{
		"Image":   "busybox",
		"Volumes": map[string]struct{}{"/tmp": {}},
	}

	status, _, err := request.SockRequest("POST", formatV123StartAPIURL("/containers/create?name="+name), config, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusCreated)

	bindPath1 := testutil.RandomTmpDirPath("test1", testEnv.DaemonPlatform())
	bindPath2 := testutil.RandomTmpDirPath("test2", testEnv.DaemonPlatform())

	config = map[string]interface{}{
		"Binds": []string{bindPath1 + ":/tmp", bindPath2 + ":/tmp"},
	}
	status, body, err := request.SockRequest("POST", formatV123StartAPIURL("/containers/"+name+"/start"), config, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusInternalServerError)
	c.Assert(string(body), checker.Contains, "Duplicate mount point", check.Commentf("Expected failure due to duplicate bind mounts to same path, instead got: %q with error: %v", string(body), err))
}

func (s *DockerSuite) TestDeprecatedContainerAPIStartVolumesFrom(c *check.C) {
	// TODO Windows to Windows CI - Port this
	testRequires(c, DaemonIsLinux)
	volName := "voltst"
	volPath := "/tmp"

	dockerCmd(c, "run", "--name", volName, "-v", volPath, "busybox")

	name := "TestContainerAPIStartVolumesFrom"
	config := map[string]interface{}{
		"Image":   "busybox",
		"Volumes": map[string]struct{}{volPath: {}},
	}

	status, _, err := request.SockRequest("POST", formatV123StartAPIURL("/containers/create?name="+name), config, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusCreated)

	config = map[string]interface{}{
		"VolumesFrom": []string{volName},
	}
	status, _, err = request.SockRequest("POST", formatV123StartAPIURL("/containers/"+name+"/start"), config, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusNoContent)

	pth, err := inspectMountSourceField(name, volPath)
	c.Assert(err, checker.IsNil)
	pth2, err := inspectMountSourceField(volName, volPath)
	c.Assert(err, checker.IsNil)
	c.Assert(pth, checker.Equals, pth2, check.Commentf("expected volume host path to be %s, got %s", pth, pth2))
}

// #9981 - Allow a docker created volume (ie, one in /var/lib/docker/volumes) to be used to overwrite (via passing in Binds on api start) an existing volume
func (s *DockerSuite) TestDeprecatedPostContainerBindNormalVolume(c *check.C) {
	// TODO Windows to Windows CI - Port this
	testRequires(c, DaemonIsLinux)
	dockerCmd(c, "create", "-v", "/foo", "--name=one", "busybox")

	fooDir, err := inspectMountSourceField("one", "/foo")
	c.Assert(err, checker.IsNil)

	dockerCmd(c, "create", "-v", "/foo", "--name=two", "busybox")

	bindSpec := map[string][]string{"Binds": {fooDir + ":/foo"}}
	status, _, err := request.SockRequest("POST", formatV123StartAPIURL("/containers/two/start"), bindSpec, daemonHost())
	c.Assert(err, checker.IsNil)
	c.Assert(status, checker.Equals, http.StatusNoContent)

	fooDir2, err := inspectMountSourceField("two", "/foo")
	c.Assert(err, checker.IsNil)
	c.Assert(fooDir2, checker.Equals, fooDir, check.Commentf("expected volume path to be %s, got: %s", fooDir, fooDir2))
}

func (s *DockerSuite) TestDeprecatedStartWithTooLowMemoryLimit(c *check.C) {
	// TODO Windows: Port once memory is supported
	testRequires(c, DaemonIsLinux)
	out, _ := dockerCmd(c, "create", "busybox")

	containerID := strings.TrimSpace(out)

	config := `{
                "CpuShares": 100,
                "Memory":    524287
        }`

	res, body, err := request.Post(formatV123StartAPIURL("/containers/"+containerID+"/start"), request.RawString(config), request.JSON)
	c.Assert(err, checker.IsNil)
	b, err2 := testutil.ReadBody(body)
	c.Assert(err2, checker.IsNil)
	c.Assert(res.StatusCode, checker.Equals, http.StatusInternalServerError)
	c.Assert(string(b), checker.Contains, "Minimum memory limit allowed is 4MB")
}

// #14640
func (s *DockerSuite) TestDeprecatedPostContainersStartWithoutLinksInHostConfig(c *check.C) {
	// TODO Windows: Windows doesn't support supplying a hostconfig on start.
	// An alternate test could be written to validate the negative testing aspect of this
	testRequires(c, DaemonIsLinux)
	name := "test-host-config-links"
	dockerCmd(c, append([]string{"create", "--name", name, "busybox"}, sleepCommandForDaemonPlatform()...)...)

	hc := inspectFieldJSON(c, name, "HostConfig")
	config := `{"HostConfig":` + hc + `}`

	res, b, err := request.Post(formatV123StartAPIURL("/containers/"+name+"/start"), request.RawString(config), request.JSON)
	c.Assert(err, checker.IsNil)
	c.Assert(res.StatusCode, checker.Equals, http.StatusNoContent)
	b.Close()
}

// #14640
func (s *DockerSuite) TestDeprecatedPostContainersStartWithLinksInHostConfig(c *check.C) {
	// TODO Windows: Windows doesn't support supplying a hostconfig on start.
	// An alternate test could be written to validate the negative testing aspect of this
	testRequires(c, DaemonIsLinux)
	name := "test-host-config-links"
	dockerCmd(c, "run", "--name", "foo", "-d", "busybox", "top")
	dockerCmd(c, "create", "--name", name, "--link", "foo:bar", "busybox", "top")

	hc := inspectFieldJSON(c, name, "HostConfig")
	config := `{"HostConfig":` + hc + `}`

	res, b, err := request.Post(formatV123StartAPIURL("/containers/"+name+"/start"), request.RawString(config), request.JSON)
	c.Assert(err, checker.IsNil)
	c.Assert(res.StatusCode, checker.Equals, http.StatusNoContent)
	b.Close()
}

// #14640
func (s *DockerSuite) TestDeprecatedPostContainersStartWithLinksInHostConfigIdLinked(c *check.C) {
	// Windows does not support links
	testRequires(c, DaemonIsLinux)
	name := "test-host-config-links"
	out, _ := dockerCmd(c, "run", "--name", "link0", "-d", "busybox", "top")
	id := strings.TrimSpace(out)
	dockerCmd(c, "create", "--name", name, "--link", id, "busybox", "top")

	hc := inspectFieldJSON(c, name, "HostConfig")
	config := `{"HostConfig":` + hc + `}`

	res, b, err := request.Post(formatV123StartAPIURL("/containers/"+name+"/start"), request.RawString(config), request.JSON)
	c.Assert(err, checker.IsNil)
	c.Assert(res.StatusCode, checker.Equals, http.StatusNoContent)
	b.Close()
}

func (s *DockerSuite) TestDeprecatedStartWithNilDNS(c *check.C) {
	// TODO Windows: Add once DNS is supported
	testRequires(c, DaemonIsLinux)
	out, _ := dockerCmd(c, "create", "busybox")
	containerID := strings.TrimSpace(out)

	config := `{"HostConfig": {"Dns": null}}`

	res, b, err := request.Post(formatV123StartAPIURL("/containers/"+containerID+"/start"), request.RawString(config), request.JSON)
	c.Assert(err, checker.IsNil)
	c.Assert(res.StatusCode, checker.Equals, http.StatusNoContent)
	b.Close()

	dns := inspectFieldJSON(c, containerID, "HostConfig.Dns")
	c.Assert(dns, checker.Equals, "[]")
}
