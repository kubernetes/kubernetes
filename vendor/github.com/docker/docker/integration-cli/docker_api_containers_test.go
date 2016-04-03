package main

import (
	"archive/tar"
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httputil"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/docker/runconfig"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestContainerApiGetAll(c *check.C) {
	startCount, err := getContainerCount()
	if err != nil {
		c.Fatalf("Cannot query container count: %v", err)
	}

	name := "getall"
	dockerCmd(c, "run", "--name", name, "busybox", "true")

	status, body, err := sockRequest("GET", "/containers/json?all=1", nil)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusOK)

	var inspectJSON []struct {
		Names []string
	}
	if err = json.Unmarshal(body, &inspectJSON); err != nil {
		c.Fatalf("unable to unmarshal response body: %v", err)
	}

	if len(inspectJSON) != startCount+1 {
		c.Fatalf("Expected %d container(s), %d found (started with: %d)", startCount+1, len(inspectJSON), startCount)
	}

	if actual := inspectJSON[0].Names[0]; actual != "/"+name {
		c.Fatalf("Container Name mismatch. Expected: %q, received: %q\n", "/"+name, actual)
	}
}

// regression test for empty json field being omitted #13691
func (s *DockerSuite) TestContainerApiGetJSONNoFieldsOmitted(c *check.C) {
	dockerCmd(c, "run", "busybox", "true")

	status, body, err := sockRequest("GET", "/containers/json?all=1", nil)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusOK)

	// empty Labels field triggered this bug, make sense to check for everything
	// cause even Ports for instance can trigger this bug
	// better safe than sorry..
	fields := []string{
		"Id",
		"Names",
		"Image",
		"Command",
		"Created",
		"Ports",
		"Labels",
		"Status",
	}

	// decoding into types.Container do not work since it eventually unmarshal
	// and empty field to an empty go map, so we just check for a string
	for _, f := range fields {
		if !strings.Contains(string(body), f) {
			c.Fatalf("Field %s is missing and it shouldn't", f)
		}
	}
}

type containerPs struct {
	Names []string
	Ports []map[string]interface{}
}

// regression test for non-empty fields from #13901
func (s *DockerSuite) TestContainerPsOmitFields(c *check.C) {
	name := "pstest"
	port := 80
	dockerCmd(c, "run", "-d", "--name", name, "--expose", strconv.Itoa(port), "busybox", "top")

	status, body, err := sockRequest("GET", "/containers/json?all=1", nil)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusOK)

	var resp []containerPs
	err = json.Unmarshal(body, &resp)
	c.Assert(err, check.IsNil)

	var foundContainer *containerPs
	for _, container := range resp {
		for _, testName := range container.Names {
			if "/"+name == testName {
				foundContainer = &container
				break
			}
		}
	}

	c.Assert(len(foundContainer.Ports), check.Equals, 1)
	c.Assert(foundContainer.Ports[0]["PrivatePort"], check.Equals, float64(port))
	_, ok := foundContainer.Ports[0]["PublicPort"]
	c.Assert(ok, check.Not(check.Equals), true)
	_, ok = foundContainer.Ports[0]["IP"]
	c.Assert(ok, check.Not(check.Equals), true)
}

func (s *DockerSuite) TestContainerApiGetExport(c *check.C) {
	name := "exportcontainer"
	dockerCmd(c, "run", "--name", name, "busybox", "touch", "/test")

	status, body, err := sockRequest("GET", "/containers/"+name+"/export", nil)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusOK)

	found := false
	for tarReader := tar.NewReader(bytes.NewReader(body)); ; {
		h, err := tarReader.Next()
		if err != nil {
			if err == io.EOF {
				break
			}
			c.Fatal(err)
		}
		if h.Name == "test" {
			found = true
			break
		}
	}

	if !found {
		c.Fatalf("The created test file has not been found in the exported image")
	}
}

func (s *DockerSuite) TestContainerApiGetChanges(c *check.C) {
	name := "changescontainer"
	dockerCmd(c, "run", "--name", name, "busybox", "rm", "/etc/passwd")

	status, body, err := sockRequest("GET", "/containers/"+name+"/changes", nil)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusOK)

	changes := []struct {
		Kind int
		Path string
	}{}
	if err = json.Unmarshal(body, &changes); err != nil {
		c.Fatalf("unable to unmarshal response body: %v", err)
	}

	// Check the changelog for removal of /etc/passwd
	success := false
	for _, elem := range changes {
		if elem.Path == "/etc/passwd" && elem.Kind == 2 {
			success = true
		}
	}
	if !success {
		c.Fatalf("/etc/passwd has been removed but is not present in the diff")
	}
}

func (s *DockerSuite) TestContainerApiStartVolumeBinds(c *check.C) {
	name := "testing"
	config := map[string]interface{}{
		"Image":   "busybox",
		"Volumes": map[string]struct{}{"/tmp": {}},
	}

	status, _, err := sockRequest("POST", "/containers/create?name="+name, config)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusCreated)

	bindPath := randomUnixTmpDirPath("test")
	config = map[string]interface{}{
		"Binds": []string{bindPath + ":/tmp"},
	}
	status, _, err = sockRequest("POST", "/containers/"+name+"/start", config)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusNoContent)

	pth, err := inspectMountSourceField(name, "/tmp")
	if err != nil {
		c.Fatal(err)
	}

	if pth != bindPath {
		c.Fatalf("expected volume host path to be %s, got %s", bindPath, pth)
	}
}

// Test for GH#10618
func (s *DockerSuite) TestContainerApiStartDupVolumeBinds(c *check.C) {
	name := "testdups"
	config := map[string]interface{}{
		"Image":   "busybox",
		"Volumes": map[string]struct{}{"/tmp": {}},
	}

	status, _, err := sockRequest("POST", "/containers/create?name="+name, config)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusCreated)

	bindPath1 := randomUnixTmpDirPath("test1")
	bindPath2 := randomUnixTmpDirPath("test2")

	config = map[string]interface{}{
		"Binds": []string{bindPath1 + ":/tmp", bindPath2 + ":/tmp"},
	}
	status, body, err := sockRequest("POST", "/containers/"+name+"/start", config)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusInternalServerError)

	if !strings.Contains(string(body), "Duplicate bind") {
		c.Fatalf("Expected failure due to duplicate bind mounts to same path, instead got: %q with error: %v", string(body), err)
	}
}

func (s *DockerSuite) TestContainerApiStartVolumesFrom(c *check.C) {
	volName := "voltst"
	volPath := "/tmp"

	dockerCmd(c, "run", "-d", "--name", volName, "-v", volPath, "busybox")

	name := "TestContainerApiStartVolumesFrom"
	config := map[string]interface{}{
		"Image":   "busybox",
		"Volumes": map[string]struct{}{volPath: {}},
	}

	status, _, err := sockRequest("POST", "/containers/create?name="+name, config)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusCreated)

	config = map[string]interface{}{
		"VolumesFrom": []string{volName},
	}
	status, _, err = sockRequest("POST", "/containers/"+name+"/start", config)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusNoContent)

	pth, err := inspectMountSourceField(name, volPath)
	if err != nil {
		c.Fatal(err)
	}
	pth2, err := inspectMountSourceField(volName, volPath)
	if err != nil {
		c.Fatal(err)
	}

	if pth != pth2 {
		c.Fatalf("expected volume host path to be %s, got %s", pth, pth2)
	}
}

func (s *DockerSuite) TestGetContainerStats(c *check.C) {
	var (
		name = "statscontainer"
	)
	dockerCmd(c, "run", "-d", "--name", name, "busybox", "top")

	type b struct {
		status int
		body   []byte
		err    error
	}
	bc := make(chan b, 1)
	go func() {
		status, body, err := sockRequest("GET", "/containers/"+name+"/stats", nil)
		bc <- b{status, body, err}
	}()

	// allow some time to stream the stats from the container
	time.Sleep(4 * time.Second)
	dockerCmd(c, "rm", "-f", name)

	// collect the results from the stats stream or timeout and fail
	// if the stream was not disconnected.
	select {
	case <-time.After(2 * time.Second):
		c.Fatal("stream was not closed after container was removed")
	case sr := <-bc:
		c.Assert(sr.err, check.IsNil)
		c.Assert(sr.status, check.Equals, http.StatusOK)

		dec := json.NewDecoder(bytes.NewBuffer(sr.body))
		var s *types.Stats
		// decode only one object from the stream
		if err := dec.Decode(&s); err != nil {
			c.Fatal(err)
		}
	}
}

func (s *DockerSuite) TestGetContainerStatsRmRunning(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "top")
	id := strings.TrimSpace(out)

	buf := &channelBuffer{make(chan []byte, 1)}
	defer buf.Close()
	chErr := make(chan error)
	go func() {
		_, body, err := sockRequestRaw("GET", "/containers/"+id+"/stats?stream=1", nil, "application/json")
		if err != nil {
			chErr <- err
		}
		defer body.Close()
		_, err = io.Copy(buf, body)
		chErr <- err
	}()
	defer func() {
		c.Assert(<-chErr, check.IsNil)
	}()

	b := make([]byte, 32)
	// make sure we've got some stats
	_, err := buf.ReadTimeout(b, 2*time.Second)
	c.Assert(err, check.IsNil)

	// Now remove without `-f` and make sure we are still pulling stats
	_, _, err = dockerCmdWithError(c, "rm", id)
	c.Assert(err, check.Not(check.IsNil), check.Commentf("rm should have failed but didn't"))
	_, err = buf.ReadTimeout(b, 2*time.Second)
	c.Assert(err, check.IsNil)
	dockerCmd(c, "rm", "-f", id)

	_, err = buf.ReadTimeout(b, 2*time.Second)
	c.Assert(err, check.Not(check.IsNil))
}

// regression test for gh13421
// previous test was just checking one stat entry so it didn't fail (stats with
// stream false always return one stat)
func (s *DockerSuite) TestGetContainerStatsStream(c *check.C) {
	name := "statscontainer"
	dockerCmd(c, "run", "-d", "--name", name, "busybox", "top")

	type b struct {
		status int
		body   []byte
		err    error
	}
	bc := make(chan b, 1)
	go func() {
		status, body, err := sockRequest("GET", "/containers/"+name+"/stats", nil)
		bc <- b{status, body, err}
	}()

	// allow some time to stream the stats from the container
	time.Sleep(4 * time.Second)
	dockerCmd(c, "rm", "-f", name)

	// collect the results from the stats stream or timeout and fail
	// if the stream was not disconnected.
	select {
	case <-time.After(2 * time.Second):
		c.Fatal("stream was not closed after container was removed")
	case sr := <-bc:
		c.Assert(sr.err, check.IsNil)
		c.Assert(sr.status, check.Equals, http.StatusOK)

		s := string(sr.body)
		// count occurrences of "read" of types.Stats
		if l := strings.Count(s, "read"); l < 2 {
			c.Fatalf("Expected more than one stat streamed, got %d", l)
		}
	}
}

func (s *DockerSuite) TestGetContainerStatsNoStream(c *check.C) {
	name := "statscontainer"
	dockerCmd(c, "run", "-d", "--name", name, "busybox", "top")

	type b struct {
		status int
		body   []byte
		err    error
	}
	bc := make(chan b, 1)
	go func() {
		status, body, err := sockRequest("GET", "/containers/"+name+"/stats?stream=0", nil)
		bc <- b{status, body, err}
	}()

	// allow some time to stream the stats from the container
	time.Sleep(4 * time.Second)
	dockerCmd(c, "rm", "-f", name)

	// collect the results from the stats stream or timeout and fail
	// if the stream was not disconnected.
	select {
	case <-time.After(2 * time.Second):
		c.Fatal("stream was not closed after container was removed")
	case sr := <-bc:
		c.Assert(sr.err, check.IsNil)
		c.Assert(sr.status, check.Equals, http.StatusOK)

		s := string(sr.body)
		// count occurrences of "read" of types.Stats
		if l := strings.Count(s, "read"); l != 1 {
			c.Fatalf("Expected only one stat streamed, got %d", l)
		}
	}
}

func (s *DockerSuite) TestGetStoppedContainerStats(c *check.C) {
	// TODO: this test does nothing because we are c.Assert'ing in goroutine
	var (
		name = "statscontainer"
	)
	dockerCmd(c, "create", "--name", name, "busybox", "top")

	go func() {
		// We'll never get return for GET stats from sockRequest as of now,
		// just send request and see if panic or error would happen on daemon side.
		status, _, err := sockRequest("GET", "/containers/"+name+"/stats", nil)
		c.Assert(err, check.IsNil)
		c.Assert(status, check.Equals, http.StatusOK)
	}()

	// allow some time to send request and let daemon deal with it
	time.Sleep(1 * time.Second)
}

func (s *DockerSuite) TestBuildApiDockerfilePath(c *check.C) {
	// Test to make sure we stop people from trying to leave the
	// build context when specifying the path to the dockerfile
	buffer := new(bytes.Buffer)
	tw := tar.NewWriter(buffer)
	defer tw.Close()

	dockerfile := []byte("FROM busybox")
	if err := tw.WriteHeader(&tar.Header{
		Name: "Dockerfile",
		Size: int64(len(dockerfile)),
	}); err != nil {
		c.Fatalf("failed to write tar file header: %v", err)
	}
	if _, err := tw.Write(dockerfile); err != nil {
		c.Fatalf("failed to write tar file content: %v", err)
	}
	if err := tw.Close(); err != nil {
		c.Fatalf("failed to close tar archive: %v", err)
	}

	res, body, err := sockRequestRaw("POST", "/build?dockerfile=../Dockerfile", buffer, "application/x-tar")
	c.Assert(err, check.IsNil)
	c.Assert(res.StatusCode, check.Equals, http.StatusInternalServerError)

	out, err := readBody(body)
	if err != nil {
		c.Fatal(err)
	}

	if !strings.Contains(string(out), "must be within the build context") {
		c.Fatalf("Didn't complain about leaving build context: %s", out)
	}
}

func (s *DockerSuite) TestBuildApiDockerFileRemote(c *check.C) {
	server, err := fakeStorage(map[string]string{
		"testD": `FROM busybox
COPY * /tmp/
RUN find / -name ba*
RUN find /tmp/`,
	})
	if err != nil {
		c.Fatal(err)
	}
	defer server.Close()

	res, body, err := sockRequestRaw("POST", "/build?dockerfile=baz&remote="+server.URL()+"/testD", nil, "application/json")
	c.Assert(err, check.IsNil)
	c.Assert(res.StatusCode, check.Equals, http.StatusOK)

	buf, err := readBody(body)
	if err != nil {
		c.Fatal(err)
	}

	// Make sure Dockerfile exists.
	// Make sure 'baz' doesn't exist ANYWHERE despite being mentioned in the URL
	out := string(buf)
	if !strings.Contains(out, "/tmp/Dockerfile") ||
		strings.Contains(out, "baz") {
		c.Fatalf("Incorrect output: %s", out)
	}
}

func (s *DockerSuite) TestBuildApiRemoteTarballContext(c *check.C) {
	buffer := new(bytes.Buffer)
	tw := tar.NewWriter(buffer)
	defer tw.Close()

	dockerfile := []byte("FROM busybox")
	if err := tw.WriteHeader(&tar.Header{
		Name: "Dockerfile",
		Size: int64(len(dockerfile)),
	}); err != nil {
		c.Fatalf("failed to write tar file header: %v", err)
	}
	if _, err := tw.Write(dockerfile); err != nil {
		c.Fatalf("failed to write tar file content: %v", err)
	}
	if err := tw.Close(); err != nil {
		c.Fatalf("failed to close tar archive: %v", err)
	}

	server, err := fakeBinaryStorage(map[string]*bytes.Buffer{
		"testT.tar": buffer,
	})
	c.Assert(err, check.IsNil)

	defer server.Close()

	res, _, err := sockRequestRaw("POST", "/build?remote="+server.URL()+"/testT.tar", nil, "application/tar")
	c.Assert(err, check.IsNil)
	c.Assert(res.StatusCode, check.Equals, http.StatusOK)
}

func (s *DockerSuite) TestBuildApiRemoteTarballContextWithCustomDockerfile(c *check.C) {
	buffer := new(bytes.Buffer)
	tw := tar.NewWriter(buffer)
	defer tw.Close()

	dockerfile := []byte(`FROM busybox
RUN echo 'wrong'`)
	if err := tw.WriteHeader(&tar.Header{
		Name: "Dockerfile",
		Size: int64(len(dockerfile)),
	}); err != nil {
		c.Fatalf("failed to write tar file header: %v", err)
	}
	if _, err := tw.Write(dockerfile); err != nil {
		c.Fatalf("failed to write tar file content: %v", err)
	}

	custom := []byte(`FROM busybox
RUN echo 'right'
`)
	if err := tw.WriteHeader(&tar.Header{
		Name: "custom",
		Size: int64(len(custom)),
	}); err != nil {
		c.Fatalf("failed to write tar file header: %v", err)
	}
	if _, err := tw.Write(custom); err != nil {
		c.Fatalf("failed to write tar file content: %v", err)
	}

	if err := tw.Close(); err != nil {
		c.Fatalf("failed to close tar archive: %v", err)
	}

	server, err := fakeBinaryStorage(map[string]*bytes.Buffer{
		"testT.tar": buffer,
	})
	c.Assert(err, check.IsNil)

	defer server.Close()
	url := "/build?dockerfile=custom&remote=" + server.URL() + "/testT.tar"
	res, body, err := sockRequestRaw("POST", url, nil, "application/tar")
	c.Assert(err, check.IsNil)
	c.Assert(res.StatusCode, check.Equals, http.StatusOK)

	defer body.Close()
	content, err := readBody(body)
	c.Assert(err, check.IsNil)

	if strings.Contains(string(content), "wrong") {
		c.Fatalf("Build used the wrong dockerfile.")
	}
}

func (s *DockerSuite) TestBuildApiLowerDockerfile(c *check.C) {
	git, err := fakeGIT("repo", map[string]string{
		"dockerfile": `FROM busybox
RUN echo from dockerfile`,
	}, false)
	if err != nil {
		c.Fatal(err)
	}
	defer git.Close()

	res, body, err := sockRequestRaw("POST", "/build?remote="+git.RepoURL, nil, "application/json")
	c.Assert(err, check.IsNil)
	c.Assert(res.StatusCode, check.Equals, http.StatusOK)

	buf, err := readBody(body)
	if err != nil {
		c.Fatal(err)
	}

	out := string(buf)
	if !strings.Contains(out, "from dockerfile") {
		c.Fatalf("Incorrect output: %s", out)
	}
}

func (s *DockerSuite) TestBuildApiBuildGitWithF(c *check.C) {
	git, err := fakeGIT("repo", map[string]string{
		"baz": `FROM busybox
RUN echo from baz`,
		"Dockerfile": `FROM busybox
RUN echo from Dockerfile`,
	}, false)
	if err != nil {
		c.Fatal(err)
	}
	defer git.Close()

	// Make sure it tries to 'dockerfile' query param value
	res, body, err := sockRequestRaw("POST", "/build?dockerfile=baz&remote="+git.RepoURL, nil, "application/json")
	c.Assert(err, check.IsNil)
	c.Assert(res.StatusCode, check.Equals, http.StatusOK)

	buf, err := readBody(body)
	if err != nil {
		c.Fatal(err)
	}

	out := string(buf)
	if !strings.Contains(out, "from baz") {
		c.Fatalf("Incorrect output: %s", out)
	}
}

func (s *DockerSuite) TestBuildApiDoubleDockerfile(c *check.C) {
	testRequires(c, UnixCli) // dockerfile overwrites Dockerfile on Windows
	git, err := fakeGIT("repo", map[string]string{
		"Dockerfile": `FROM busybox
RUN echo from Dockerfile`,
		"dockerfile": `FROM busybox
RUN echo from dockerfile`,
	}, false)
	if err != nil {
		c.Fatal(err)
	}
	defer git.Close()

	// Make sure it tries to 'dockerfile' query param value
	res, body, err := sockRequestRaw("POST", "/build?remote="+git.RepoURL, nil, "application/json")
	c.Assert(err, check.IsNil)
	c.Assert(res.StatusCode, check.Equals, http.StatusOK)

	buf, err := readBody(body)
	if err != nil {
		c.Fatal(err)
	}

	out := string(buf)
	if !strings.Contains(out, "from Dockerfile") {
		c.Fatalf("Incorrect output: %s", out)
	}
}

func (s *DockerSuite) TestBuildApiDockerfileSymlink(c *check.C) {
	// Test to make sure we stop people from trying to leave the
	// build context when specifying a symlink as the path to the dockerfile
	buffer := new(bytes.Buffer)
	tw := tar.NewWriter(buffer)
	defer tw.Close()

	if err := tw.WriteHeader(&tar.Header{
		Name:     "Dockerfile",
		Typeflag: tar.TypeSymlink,
		Linkname: "/etc/passwd",
	}); err != nil {
		c.Fatalf("failed to write tar file header: %v", err)
	}
	if err := tw.Close(); err != nil {
		c.Fatalf("failed to close tar archive: %v", err)
	}

	res, body, err := sockRequestRaw("POST", "/build", buffer, "application/x-tar")
	c.Assert(err, check.IsNil)
	c.Assert(res.StatusCode, check.Equals, http.StatusInternalServerError)

	out, err := readBody(body)
	if err != nil {
		c.Fatal(err)
	}

	// The reason the error is "Cannot locate specified Dockerfile" is because
	// in the builder, the symlink is resolved within the context, therefore
	// Dockerfile -> /etc/passwd becomes etc/passwd from the context which is
	// a nonexistent file.
	if !strings.Contains(string(out), "Cannot locate specified Dockerfile: Dockerfile") {
		c.Fatalf("Didn't complain about leaving build context: %s", out)
	}
}

// #9981 - Allow a docker created volume (ie, one in /var/lib/docker/volumes) to be used to overwrite (via passing in Binds on api start) an existing volume
func (s *DockerSuite) TestPostContainerBindNormalVolume(c *check.C) {
	dockerCmd(c, "create", "-v", "/foo", "--name=one", "busybox")

	fooDir, err := inspectMountSourceField("one", "/foo")
	if err != nil {
		c.Fatal(err)
	}

	dockerCmd(c, "create", "-v", "/foo", "--name=two", "busybox")

	bindSpec := map[string][]string{"Binds": {fooDir + ":/foo"}}
	status, _, err := sockRequest("POST", "/containers/two/start", bindSpec)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusNoContent)

	fooDir2, err := inspectMountSourceField("two", "/foo")
	if err != nil {
		c.Fatal(err)
	}

	if fooDir2 != fooDir {
		c.Fatalf("expected volume path to be %s, got: %s", fooDir, fooDir2)
	}
}

func (s *DockerSuite) TestContainerApiPause(c *check.C) {
	defer unpauseAllContainers()
	out, _ := dockerCmd(c, "run", "-d", "busybox", "sleep", "30")
	ContainerID := strings.TrimSpace(out)

	status, _, err := sockRequest("POST", "/containers/"+ContainerID+"/pause", nil)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusNoContent)

	pausedContainers, err := getSliceOfPausedContainers()

	if err != nil {
		c.Fatalf("error thrown while checking if containers were paused: %v", err)
	}

	if len(pausedContainers) != 1 || stringid.TruncateID(ContainerID) != pausedContainers[0] {
		c.Fatalf("there should be one paused container and not %d", len(pausedContainers))
	}

	status, _, err = sockRequest("POST", "/containers/"+ContainerID+"/unpause", nil)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusNoContent)

	pausedContainers, err = getSliceOfPausedContainers()

	if err != nil {
		c.Fatalf("error thrown while checking if containers were paused: %v", err)
	}

	if pausedContainers != nil {
		c.Fatalf("There should be no paused container.")
	}
}

func (s *DockerSuite) TestContainerApiTop(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "/bin/sh", "-c", "top")
	id := strings.TrimSpace(string(out))
	if err := waitRun(id); err != nil {
		c.Fatal(err)
	}

	type topResp struct {
		Titles    []string
		Processes [][]string
	}
	var top topResp
	status, b, err := sockRequest("GET", "/containers/"+id+"/top?ps_args=aux", nil)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusOK)

	if err := json.Unmarshal(b, &top); err != nil {
		c.Fatal(err)
	}

	if len(top.Titles) != 11 {
		c.Fatalf("expected 11 titles, found %d: %v", len(top.Titles), top.Titles)
	}

	if top.Titles[0] != "USER" || top.Titles[10] != "COMMAND" {
		c.Fatalf("expected `USER` at `Titles[0]` and `COMMAND` at Titles[10]: %v", top.Titles)
	}
	if len(top.Processes) != 2 {
		c.Fatalf("expected 2 processes, found %d: %v", len(top.Processes), top.Processes)
	}
	if top.Processes[0][10] != "/bin/sh -c top" {
		c.Fatalf("expected `/bin/sh -c top`, found: %s", top.Processes[0][10])
	}
	if top.Processes[1][10] != "top" {
		c.Fatalf("expected `top`, found: %s", top.Processes[1][10])
	}
}

func (s *DockerSuite) TestContainerApiCommit(c *check.C) {
	cName := "testapicommit"
	dockerCmd(c, "run", "--name="+cName, "busybox", "/bin/sh", "-c", "touch /test")

	name := "TestContainerApiCommit"
	status, b, err := sockRequest("POST", "/commit?repo="+name+"&testtag=tag&container="+cName, nil)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusCreated)

	type resp struct {
		Id string
	}
	var img resp
	if err := json.Unmarshal(b, &img); err != nil {
		c.Fatal(err)
	}

	cmd, err := inspectField(img.Id, "Config.Cmd")
	if err != nil {
		c.Fatal(err)
	}
	if cmd != "{[/bin/sh -c touch /test]}" {
		c.Fatalf("got wrong Cmd from commit: %q", cmd)
	}
	// sanity check, make sure the image is what we think it is
	dockerCmd(c, "run", img.Id, "ls", "/test")
}

func (s *DockerSuite) TestContainerApiCommitWithLabelInConfig(c *check.C) {
	cName := "testapicommitwithconfig"
	dockerCmd(c, "run", "--name="+cName, "busybox", "/bin/sh", "-c", "touch /test")

	config := map[string]interface{}{
		"Labels": map[string]string{"key1": "value1", "key2": "value2"},
	}

	name := "TestContainerApiCommitWithConfig"
	status, b, err := sockRequest("POST", "/commit?repo="+name+"&container="+cName, config)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusCreated)

	type resp struct {
		Id string
	}
	var img resp
	if err := json.Unmarshal(b, &img); err != nil {
		c.Fatal(err)
	}

	label1, err := inspectFieldMap(img.Id, "Config.Labels", "key1")
	if err != nil {
		c.Fatal(err)
	}
	c.Assert(label1, check.Equals, "value1")

	label2, err := inspectFieldMap(img.Id, "Config.Labels", "key2")
	if err != nil {
		c.Fatal(err)
	}
	c.Assert(label2, check.Equals, "value2")

	cmd, err := inspectField(img.Id, "Config.Cmd")
	if err != nil {
		c.Fatal(err)
	}
	if cmd != "{[/bin/sh -c touch /test]}" {
		c.Fatalf("got wrong Cmd from commit: %q", cmd)
	}

	// sanity check, make sure the image is what we think it is
	dockerCmd(c, "run", img.Id, "ls", "/test")
}

func (s *DockerSuite) TestContainerApiBadPort(c *check.C) {
	config := map[string]interface{}{
		"Image": "busybox",
		"Cmd":   []string{"/bin/sh", "-c", "echo test"},
		"PortBindings": map[string]interface{}{
			"8080/tcp": []map[string]interface{}{
				{
					"HostIP":   "",
					"HostPort": "aa80",
				},
			},
		},
	}

	jsonData := bytes.NewBuffer(nil)
	json.NewEncoder(jsonData).Encode(config)

	status, b, err := sockRequest("POST", "/containers/create", config)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusInternalServerError)

	if strings.TrimSpace(string(b)) != `Invalid port specification: "aa80"` {
		c.Fatalf("Incorrect error msg: %s", string(b))
	}
}

func (s *DockerSuite) TestContainerApiCreate(c *check.C) {
	config := map[string]interface{}{
		"Image": "busybox",
		"Cmd":   []string{"/bin/sh", "-c", "touch /test && ls /test"},
	}

	status, b, err := sockRequest("POST", "/containers/create", config)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusCreated)

	type createResp struct {
		Id string
	}
	var container createResp
	if err := json.Unmarshal(b, &container); err != nil {
		c.Fatal(err)
	}

	out, _ := dockerCmd(c, "start", "-a", container.Id)
	if strings.TrimSpace(out) != "/test" {
		c.Fatalf("expected output `/test`, got %q", out)
	}
}

func (s *DockerSuite) TestContainerApiCreateEmptyConfig(c *check.C) {
	config := map[string]interface{}{}

	status, b, err := sockRequest("POST", "/containers/create", config)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusInternalServerError)

	expected := "Config cannot be empty in order to create a container\n"
	if body := string(b); body != expected {
		c.Fatalf("Expected to get %q, got %q", expected, body)
	}
}

func (s *DockerSuite) TestContainerApiCreateWithHostName(c *check.C) {
	hostName := "test-host"
	config := map[string]interface{}{
		"Image":    "busybox",
		"Hostname": hostName,
	}

	status, body, err := sockRequest("POST", "/containers/create", config)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusCreated)

	var container types.ContainerCreateResponse
	if err := json.Unmarshal(body, &container); err != nil {
		c.Fatal(err)
	}

	status, body, err = sockRequest("GET", "/containers/"+container.ID+"/json", nil)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusOK)

	var containerJSON types.ContainerJSON
	if err := json.Unmarshal(body, &containerJSON); err != nil {
		c.Fatal(err)
	}

	if containerJSON.Config.Hostname != hostName {
		c.Fatalf("Mismatched Hostname, Expected %s, Actual: %s ", hostName, containerJSON.Config.Hostname)
	}
}

func (s *DockerSuite) TestContainerApiCreateWithDomainName(c *check.C) {
	domainName := "test-domain"
	config := map[string]interface{}{
		"Image":      "busybox",
		"Domainname": domainName,
	}

	status, body, err := sockRequest("POST", "/containers/create", config)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusCreated)

	var container types.ContainerCreateResponse
	if err := json.Unmarshal(body, &container); err != nil {
		c.Fatal(err)
	}

	status, body, err = sockRequest("GET", "/containers/"+container.ID+"/json", nil)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusOK)

	var containerJSON types.ContainerJSON
	if err := json.Unmarshal(body, &containerJSON); err != nil {
		c.Fatal(err)
	}

	if containerJSON.Config.Domainname != domainName {
		c.Fatalf("Mismatched Domainname, Expected %s, Actual: %s ", domainName, containerJSON.Config.Domainname)
	}
}

func (s *DockerSuite) TestContainerApiCreateNetworkMode(c *check.C) {
	UtilCreateNetworkMode(c, "host")
	UtilCreateNetworkMode(c, "bridge")
	UtilCreateNetworkMode(c, "container:web1")
}

func UtilCreateNetworkMode(c *check.C, networkMode string) {
	config := map[string]interface{}{
		"Image":      "busybox",
		"HostConfig": map[string]interface{}{"NetworkMode": networkMode},
	}

	status, body, err := sockRequest("POST", "/containers/create", config)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusCreated)

	var container types.ContainerCreateResponse
	if err := json.Unmarshal(body, &container); err != nil {
		c.Fatal(err)
	}

	status, body, err = sockRequest("GET", "/containers/"+container.ID+"/json", nil)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusOK)

	var containerJSON types.ContainerJSON
	if err := json.Unmarshal(body, &containerJSON); err != nil {
		c.Fatal(err)
	}

	if containerJSON.HostConfig.NetworkMode != runconfig.NetworkMode(networkMode) {
		c.Fatalf("Mismatched NetworkMode, Expected %s, Actual: %s ", networkMode, containerJSON.HostConfig.NetworkMode)
	}
}

func (s *DockerSuite) TestContainerApiCreateWithCpuSharesCpuset(c *check.C) {
	config := map[string]interface{}{
		"Image":      "busybox",
		"CpuShares":  512,
		"CpusetCpus": "0,1",
	}

	status, body, err := sockRequest("POST", "/containers/create", config)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusCreated)

	var container types.ContainerCreateResponse
	if err := json.Unmarshal(body, &container); err != nil {
		c.Fatal(err)
	}

	status, body, err = sockRequest("GET", "/containers/"+container.ID+"/json", nil)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusOK)

	var containerJson types.ContainerJSON

	c.Assert(json.Unmarshal(body, &containerJson), check.IsNil)

	out, err := inspectField(containerJson.Id, "HostConfig.CpuShares")
	c.Assert(err, check.IsNil)
	c.Assert(out, check.Equals, "512")

	outCpuset, errCpuset := inspectField(containerJson.Id, "HostConfig.CpusetCpus")
	c.Assert(errCpuset, check.IsNil, check.Commentf("Output: %s", outCpuset))
	c.Assert(outCpuset, check.Equals, "0,1")
}

func (s *DockerSuite) TestContainerApiVerifyHeader(c *check.C) {
	config := map[string]interface{}{
		"Image": "busybox",
	}

	create := func(ct string) (*http.Response, io.ReadCloser, error) {
		jsonData := bytes.NewBuffer(nil)
		if err := json.NewEncoder(jsonData).Encode(config); err != nil {
			c.Fatal(err)
		}
		return sockRequestRaw("POST", "/containers/create", jsonData, ct)
	}

	// Try with no content-type
	res, body, err := create("")
	c.Assert(err, check.IsNil)
	c.Assert(res.StatusCode, check.Equals, http.StatusInternalServerError)
	body.Close()

	// Try with wrong content-type
	res, body, err = create("application/xml")
	c.Assert(err, check.IsNil)
	c.Assert(res.StatusCode, check.Equals, http.StatusInternalServerError)
	body.Close()

	// now application/json
	res, body, err = create("application/json")
	c.Assert(err, check.IsNil)
	c.Assert(res.StatusCode, check.Equals, http.StatusCreated)
	body.Close()
}

//Issue 14230. daemon should return 500 for invalid port syntax
func (s *DockerSuite) TestContainerApiInvalidPortSyntax(c *check.C) {
	config := `{
				  "Image": "busybox",
				  "HostConfig": {
					"PortBindings": {
					  "19039;1230": [
						{}
					  ]
					}
				  }
				}`

	res, body, err := sockRequestRaw("POST", "/containers/create", strings.NewReader(config), "application/json")
	c.Assert(err, check.IsNil)
	c.Assert(res.StatusCode, check.Equals, http.StatusInternalServerError)

	b, err := readBody(body)
	if err != nil {
		c.Fatal(err)
	}
	c.Assert(strings.Contains(string(b[:]), "Invalid port"), check.Equals, true)
}

// Issue 7941 - test to make sure a "null" in JSON is just ignored.
// W/o this fix a null in JSON would be parsed into a string var as "null"
func (s *DockerSuite) TestContainerApiPostCreateNull(c *check.C) {
	config := `{
		"Hostname":"",
		"Domainname":"",
		"Memory":0,
		"MemorySwap":0,
		"CpuShares":0,
		"Cpuset":null,
		"AttachStdin":true,
		"AttachStdout":true,
		"AttachStderr":true,
		"ExposedPorts":{},
		"Tty":true,
		"OpenStdin":true,
		"StdinOnce":true,
		"Env":[],
		"Cmd":"ls",
		"Image":"busybox",
		"Volumes":{},
		"WorkingDir":"",
		"Entrypoint":null,
		"NetworkDisabled":false,
		"OnBuild":null}`

	res, body, err := sockRequestRaw("POST", "/containers/create", strings.NewReader(config), "application/json")
	c.Assert(err, check.IsNil)
	c.Assert(res.StatusCode, check.Equals, http.StatusCreated)

	b, err := readBody(body)
	if err != nil {
		c.Fatal(err)
	}
	type createResp struct {
		Id string
	}
	var container createResp
	if err := json.Unmarshal(b, &container); err != nil {
		c.Fatal(err)
	}

	out, err := inspectField(container.Id, "HostConfig.CpusetCpus")
	if err != nil {
		c.Fatal(err, out)
	}
	if out != "" {
		c.Fatalf("expected empty string, got %q", out)
	}

	outMemory, errMemory := inspectField(container.Id, "HostConfig.Memory")
	c.Assert(outMemory, check.Equals, "0")
	if errMemory != nil {
		c.Fatal(errMemory, outMemory)
	}
	outMemorySwap, errMemorySwap := inspectField(container.Id, "HostConfig.MemorySwap")
	c.Assert(outMemorySwap, check.Equals, "0")
	if errMemorySwap != nil {
		c.Fatal(errMemorySwap, outMemorySwap)
	}
}

func (s *DockerSuite) TestCreateWithTooLowMemoryLimit(c *check.C) {
	config := `{
		"Image":     "busybox",
		"Cmd":       "ls",
		"OpenStdin": true,
		"CpuShares": 100,
		"Memory":    524287
	}`

	res, body, err := sockRequestRaw("POST", "/containers/create", strings.NewReader(config), "application/json")
	c.Assert(err, check.IsNil)
	b, err2 := readBody(body)
	if err2 != nil {
		c.Fatal(err2)
	}

	c.Assert(res.StatusCode, check.Equals, http.StatusInternalServerError)
	c.Assert(strings.Contains(string(b), "Minimum memory limit allowed is 4MB"), check.Equals, true)
}

func (s *DockerSuite) TestStartWithTooLowMemoryLimit(c *check.C) {
	out, _ := dockerCmd(c, "create", "busybox")

	containerID := strings.TrimSpace(out)

	config := `{
                "CpuShares": 100,
                "Memory":    524287
        }`

	res, body, err := sockRequestRaw("POST", "/containers/"+containerID+"/start", strings.NewReader(config), "application/json")
	c.Assert(err, check.IsNil)
	b, err2 := readBody(body)
	if err2 != nil {
		c.Fatal(err2)
	}

	c.Assert(res.StatusCode, check.Equals, http.StatusInternalServerError)
	c.Assert(strings.Contains(string(b), "Minimum memory limit allowed is 4MB"), check.Equals, true)
}

func (s *DockerSuite) TestContainerApiRename(c *check.C) {
	out, _ := dockerCmd(c, "run", "--name", "TestContainerApiRename", "-d", "busybox", "sh")

	containerID := strings.TrimSpace(out)
	newName := "TestContainerApiRenameNew"
	statusCode, _, err := sockRequest("POST", "/containers/"+containerID+"/rename?name="+newName, nil)
	c.Assert(err, check.IsNil)
	// 204 No Content is expected, not 200
	c.Assert(statusCode, check.Equals, http.StatusNoContent)

	name, err := inspectField(containerID, "Name")
	if name != "/"+newName {
		c.Fatalf("Failed to rename container, expected %v, got %v. Container rename API failed", newName, name)
	}
}

func (s *DockerSuite) TestContainerApiKill(c *check.C) {
	name := "test-api-kill"
	dockerCmd(c, "run", "-di", "--name", name, "busybox", "top")

	status, _, err := sockRequest("POST", "/containers/"+name+"/kill", nil)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusNoContent)

	state, err := inspectField(name, "State.Running")
	if err != nil {
		c.Fatal(err)
	}
	if state != "false" {
		c.Fatalf("got wrong State from container %s: %q", name, state)
	}
}

func (s *DockerSuite) TestContainerApiRestart(c *check.C) {
	name := "test-api-restart"
	dockerCmd(c, "run", "-di", "--name", name, "busybox", "top")

	status, _, err := sockRequest("POST", "/containers/"+name+"/restart?t=1", nil)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusNoContent)

	if err := waitInspect(name, "{{ .State.Restarting  }} {{ .State.Running  }}", "false true", 5); err != nil {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestContainerApiRestartNotimeoutParam(c *check.C) {
	name := "test-api-restart-no-timeout-param"
	out, _ := dockerCmd(c, "run", "-di", "--name", name, "busybox", "top")
	id := strings.TrimSpace(out)
	c.Assert(waitRun(id), check.IsNil)

	status, _, err := sockRequest("POST", "/containers/"+name+"/restart", nil)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusNoContent)

	if err := waitInspect(name, "{{ .State.Restarting  }} {{ .State.Running  }}", "false true", 5); err != nil {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestContainerApiStart(c *check.C) {
	name := "testing-start"
	config := map[string]interface{}{
		"Image":     "busybox",
		"Cmd":       []string{"/bin/sh", "-c", "/bin/top"},
		"OpenStdin": true,
	}

	status, _, err := sockRequest("POST", "/containers/create?name="+name, config)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusCreated)

	conf := make(map[string]interface{})
	status, _, err = sockRequest("POST", "/containers/"+name+"/start", conf)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusNoContent)

	// second call to start should give 304
	status, _, err = sockRequest("POST", "/containers/"+name+"/start", conf)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusNotModified)
}

func (s *DockerSuite) TestContainerApiStop(c *check.C) {
	name := "test-api-stop"
	dockerCmd(c, "run", "-di", "--name", name, "busybox", "top")

	status, _, err := sockRequest("POST", "/containers/"+name+"/stop?t=1", nil)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusNoContent)

	if err := waitInspect(name, "{{ .State.Running  }}", "false", 5); err != nil {
		c.Fatal(err)
	}

	// second call to start should give 304
	status, _, err = sockRequest("POST", "/containers/"+name+"/stop?t=1", nil)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusNotModified)
}

func (s *DockerSuite) TestContainerApiWait(c *check.C) {
	name := "test-api-wait"
	dockerCmd(c, "run", "--name", name, "busybox", "sleep", "5")

	status, body, err := sockRequest("POST", "/containers/"+name+"/wait", nil)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusOK)

	if err := waitInspect(name, "{{ .State.Running  }}", "false", 5); err != nil {
		c.Fatal(err)
	}

	var waitres types.ContainerWaitResponse
	if err := json.Unmarshal(body, &waitres); err != nil {
		c.Fatalf("unable to unmarshal response body: %v", err)
	}

	if waitres.StatusCode != 0 {
		c.Fatalf("Expected wait response StatusCode to be 0, got %d", waitres.StatusCode)
	}
}

func (s *DockerSuite) TestContainerApiCopy(c *check.C) {
	name := "test-container-api-copy"
	dockerCmd(c, "run", "--name", name, "busybox", "touch", "/test.txt")

	postData := types.CopyConfig{
		Resource: "/test.txt",
	}

	status, body, err := sockRequest("POST", "/containers/"+name+"/copy", postData)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusOK)

	found := false
	for tarReader := tar.NewReader(bytes.NewReader(body)); ; {
		h, err := tarReader.Next()
		if err != nil {
			if err == io.EOF {
				break
			}
			c.Fatal(err)
		}
		if h.Name == "test.txt" {
			found = true
			break
		}
	}
	c.Assert(found, check.Equals, true)
}

func (s *DockerSuite) TestContainerApiCopyResourcePathEmpty(c *check.C) {
	name := "test-container-api-copy-resource-empty"
	dockerCmd(c, "run", "--name", name, "busybox", "touch", "/test.txt")

	postData := types.CopyConfig{
		Resource: "",
	}

	status, body, err := sockRequest("POST", "/containers/"+name+"/copy", postData)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusInternalServerError)
	c.Assert(string(body), check.Matches, "Path cannot be empty\n")
}

func (s *DockerSuite) TestContainerApiCopyResourcePathNotFound(c *check.C) {
	name := "test-container-api-copy-resource-not-found"
	dockerCmd(c, "run", "--name", name, "busybox")

	postData := types.CopyConfig{
		Resource: "/notexist",
	}

	status, body, err := sockRequest("POST", "/containers/"+name+"/copy", postData)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusInternalServerError)
	c.Assert(string(body), check.Matches, "Could not find the file /notexist in container "+name+"\n")
}

func (s *DockerSuite) TestContainerApiCopyContainerNotFound(c *check.C) {
	postData := types.CopyConfig{
		Resource: "/something",
	}

	status, _, err := sockRequest("POST", "/containers/notexists/copy", postData)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusNotFound)
}

func (s *DockerSuite) TestContainerApiDelete(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "top")

	id := strings.TrimSpace(out)
	c.Assert(waitRun(id), check.IsNil)

	dockerCmd(c, "stop", id)

	status, _, err := sockRequest("DELETE", "/containers/"+id, nil)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusNoContent)
}

func (s *DockerSuite) TestContainerApiDeleteNotExist(c *check.C) {
	status, body, err := sockRequest("DELETE", "/containers/doesnotexist", nil)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusNotFound)
	c.Assert(string(body), check.Matches, "no such id: doesnotexist\n")
}

func (s *DockerSuite) TestContainerApiDeleteForce(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "top")

	id := strings.TrimSpace(out)
	c.Assert(waitRun(id), check.IsNil)

	status, _, err := sockRequest("DELETE", "/containers/"+id+"?force=1", nil)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusNoContent)
}

func (s *DockerSuite) TestContainerApiDeleteRemoveLinks(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "--name", "tlink1", "busybox", "top")

	id := strings.TrimSpace(out)
	c.Assert(waitRun(id), check.IsNil)

	out, _ = dockerCmd(c, "run", "--link", "tlink1:tlink1", "--name", "tlink2", "-d", "busybox", "top")

	id2 := strings.TrimSpace(out)
	c.Assert(waitRun(id2), check.IsNil)

	links, err := inspectFieldJSON(id2, "HostConfig.Links")
	c.Assert(err, check.IsNil)

	if links != "[\"/tlink1:/tlink2/tlink1\"]" {
		c.Fatal("expected to have links between containers")
	}

	status, _, err := sockRequest("DELETE", "/containers/tlink2/tlink1?link=1", nil)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusNoContent)

	linksPostRm, err := inspectFieldJSON(id2, "HostConfig.Links")
	c.Assert(err, check.IsNil)

	if linksPostRm != "null" {
		c.Fatal("call to api deleteContainer links should have removed the specified links")
	}
}

func (s *DockerSuite) TestContainerApiDeleteConflict(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "top")

	id := strings.TrimSpace(out)
	c.Assert(waitRun(id), check.IsNil)

	status, _, err := sockRequest("DELETE", "/containers/"+id, nil)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusConflict)
}

func (s *DockerSuite) TestContainerApiDeleteRemoveVolume(c *check.C) {
	testRequires(c, SameHostDaemon)

	out, _ := dockerCmd(c, "run", "-d", "-v", "/testvolume", "busybox", "top")

	id := strings.TrimSpace(out)
	c.Assert(waitRun(id), check.IsNil)

	source, err := inspectMountSourceField(id, "/testvolume")
	_, err = os.Stat(source)
	c.Assert(err, check.IsNil)

	status, _, err := sockRequest("DELETE", "/containers/"+id+"?v=1&force=1", nil)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusNoContent)

	if _, err := os.Stat(source); !os.IsNotExist(err) {
		c.Fatalf("expected to get ErrNotExist error, got %v", err)
	}
}

// Regression test for https://github.com/docker/docker/issues/6231
func (s *DockerSuite) TestContainersApiChunkedEncoding(c *check.C) {
	out, _ := dockerCmd(c, "create", "-v", "/foo", "busybox", "true")
	id := strings.TrimSpace(out)

	conn, err := sockConn(time.Duration(10 * time.Second))
	if err != nil {
		c.Fatal(err)
	}
	client := httputil.NewClientConn(conn, nil)
	defer client.Close()

	bindCfg := strings.NewReader(`{"Binds": ["/tmp:/foo"]}`)
	req, err := http.NewRequest("POST", "/containers/"+id+"/start", bindCfg)
	if err != nil {
		c.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	// This is a cheat to make the http request do chunked encoding
	// Otherwise (just setting the Content-Encoding to chunked) net/http will overwrite
	// https://golang.org/src/pkg/net/http/request.go?s=11980:12172
	req.ContentLength = -1

	resp, err := client.Do(req)
	if err != nil {
		c.Fatalf("error starting container with chunked encoding: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 204 {
		c.Fatalf("expected status code 204, got %d", resp.StatusCode)
	}

	out, err = inspectFieldJSON(id, "HostConfig.Binds")
	if err != nil {
		c.Fatal(err)
	}

	var binds []string
	if err := json.NewDecoder(strings.NewReader(out)).Decode(&binds); err != nil {
		c.Fatal(err)
	}
	if len(binds) != 1 {
		c.Fatalf("got unexpected binds: %v", binds)
	}

	expected := "/tmp:/foo"
	if binds[0] != expected {
		c.Fatalf("got incorrect bind spec, wanted %s, got: %s", expected, binds[0])
	}
}

func (s *DockerSuite) TestPostContainerStop(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "top")

	containerID := strings.TrimSpace(out)
	c.Assert(waitRun(containerID), check.IsNil)

	statusCode, _, err := sockRequest("POST", "/containers/"+containerID+"/stop", nil)
	c.Assert(err, check.IsNil)
	// 204 No Content is expected, not 200
	c.Assert(statusCode, check.Equals, http.StatusNoContent)

	if err := waitInspect(containerID, "{{ .State.Running  }}", "false", 5); err != nil {
		c.Fatal(err)
	}
}

// #14170
func (s *DockerSuite) TestPostContainersCreateWithStringOrSliceEntrypoint(c *check.C) {
	config := struct {
		Image      string
		Entrypoint string
		Cmd        []string
	}{"busybox", "echo", []string{"hello", "world"}}
	_, _, err := sockRequest("POST", "/containers/create?name=echotest", config)
	c.Assert(err, check.IsNil)
	out, _ := dockerCmd(c, "start", "-a", "echotest")
	c.Assert(strings.TrimSpace(out), check.Equals, "hello world")

	config2 := struct {
		Image      string
		Entrypoint []string
		Cmd        []string
	}{"busybox", []string{"echo"}, []string{"hello", "world"}}
	_, _, err = sockRequest("POST", "/containers/create?name=echotest2", config2)
	c.Assert(err, check.IsNil)
	out, _ = dockerCmd(c, "start", "-a", "echotest2")
	c.Assert(strings.TrimSpace(out), check.Equals, "hello world")
}

// #14170
func (s *DockerSuite) TestPostContainersCreateWithStringOrSliceCmd(c *check.C) {
	config := struct {
		Image      string
		Entrypoint string
		Cmd        string
	}{"busybox", "echo", "hello world"}
	_, _, err := sockRequest("POST", "/containers/create?name=echotest", config)
	c.Assert(err, check.IsNil)
	out, _ := dockerCmd(c, "start", "-a", "echotest")
	c.Assert(strings.TrimSpace(out), check.Equals, "hello world")

	config2 := struct {
		Image string
		Cmd   []string
	}{"busybox", []string{"echo", "hello", "world"}}
	_, _, err = sockRequest("POST", "/containers/create?name=echotest2", config2)
	c.Assert(err, check.IsNil)
	out, _ = dockerCmd(c, "start", "-a", "echotest2")
	c.Assert(strings.TrimSpace(out), check.Equals, "hello world")
}

// regression #14318
func (s *DockerSuite) TestPostContainersCreateWithStringOrSliceCapAddDrop(c *check.C) {
	config := struct {
		Image   string
		CapAdd  string
		CapDrop string
	}{"busybox", "NET_ADMIN", "SYS_ADMIN"}
	status, _, err := sockRequest("POST", "/containers/create?name=capaddtest0", config)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusCreated)

	config2 := struct {
		Image   string
		CapAdd  []string
		CapDrop []string
	}{"busybox", []string{"NET_ADMIN", "SYS_ADMIN"}, []string{"SETGID"}}
	status, _, err = sockRequest("POST", "/containers/create?name=capaddtest1", config2)
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusCreated)
}

// #14640
func (s *DockerSuite) TestPostContainersStartWithoutLinksInHostConfig(c *check.C) {
	name := "test-host-config-links"
	dockerCmd(c, "create", "--name", name, "busybox", "top")

	hc, err := inspectFieldJSON(name, "HostConfig")
	c.Assert(err, check.IsNil)
	config := `{"HostConfig":` + hc + `}`

	res, _, err := sockRequestRaw("POST", "/containers/"+name+"/start", strings.NewReader(config), "application/json")
	c.Assert(err, check.IsNil)
	c.Assert(res.StatusCode, check.Equals, http.StatusNoContent)
}

// #14640
func (s *DockerSuite) TestPostContainersStartWithLinksInHostConfig(c *check.C) {
	name := "test-host-config-links"
	dockerCmd(c, "run", "--name", "foo", "-d", "busybox", "top")
	dockerCmd(c, "create", "--name", name, "--link", "foo:bar", "busybox", "top")

	hc, err := inspectFieldJSON(name, "HostConfig")
	c.Assert(err, check.IsNil)
	config := `{"HostConfig":` + hc + `}`

	res, _, err := sockRequestRaw("POST", "/containers/"+name+"/start", strings.NewReader(config), "application/json")
	c.Assert(err, check.IsNil)
	c.Assert(res.StatusCode, check.Equals, http.StatusNoContent)
}

// #14640
func (s *DockerSuite) TestPostContainersStartWithLinksInHostConfigIdLinked(c *check.C) {
	name := "test-host-config-links"
	out, _ := dockerCmd(c, "run", "--name", "link0", "-d", "busybox", "top")
	id := strings.TrimSpace(out)
	dockerCmd(c, "create", "--name", name, "--link", id, "busybox", "top")

	hc, err := inspectFieldJSON(name, "HostConfig")
	c.Assert(err, check.IsNil)
	config := `{"HostConfig":` + hc + `}`

	res, _, err := sockRequestRaw("POST", "/containers/"+name+"/start", strings.NewReader(config), "application/json")
	c.Assert(err, check.IsNil)
	c.Assert(res.StatusCode, check.Equals, http.StatusNoContent)
}
