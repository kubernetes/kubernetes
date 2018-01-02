package main

import (
	"archive/tar"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"regexp"
	"strings"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/cli/build/fakecontext"
	"github.com/docker/docker/integration-cli/cli/build/fakegit"
	"github.com/docker/docker/integration-cli/cli/build/fakestorage"
	"github.com/docker/docker/integration-cli/request"
	"github.com/docker/docker/pkg/testutil"
	"github.com/go-check/check"
	"github.com/moby/buildkit/session"
	"github.com/moby/buildkit/session/filesync"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/net/context"
	"golang.org/x/sync/errgroup"
)

func (s *DockerSuite) TestBuildAPIDockerFileRemote(c *check.C) {
	testRequires(c, NotUserNamespace)
	var testD string
	if testEnv.DaemonPlatform() == "windows" {
		testD = `FROM busybox
RUN find / -name ba*
RUN find /tmp/`
	} else {
		// -xdev is required because sysfs can cause EPERM
		testD = `FROM busybox
RUN find / -xdev -name ba*
RUN find /tmp/`
	}
	server := fakestorage.New(c, "", fakecontext.WithFiles(map[string]string{"testD": testD}))
	defer server.Close()

	res, body, err := request.Post("/build?dockerfile=baz&remote="+server.URL()+"/testD", request.JSON)
	c.Assert(err, checker.IsNil)
	c.Assert(res.StatusCode, checker.Equals, http.StatusOK)

	buf, err := testutil.ReadBody(body)
	c.Assert(err, checker.IsNil)

	// Make sure Dockerfile exists.
	// Make sure 'baz' doesn't exist ANYWHERE despite being mentioned in the URL
	out := string(buf)
	c.Assert(out, checker.Contains, "RUN find /tmp")
	c.Assert(out, checker.Not(checker.Contains), "baz")
}

func (s *DockerSuite) TestBuildAPIRemoteTarballContext(c *check.C) {
	buffer := new(bytes.Buffer)
	tw := tar.NewWriter(buffer)
	defer tw.Close()

	dockerfile := []byte("FROM busybox")
	err := tw.WriteHeader(&tar.Header{
		Name: "Dockerfile",
		Size: int64(len(dockerfile)),
	})
	// failed to write tar file header
	c.Assert(err, checker.IsNil)

	_, err = tw.Write(dockerfile)
	// failed to write tar file content
	c.Assert(err, checker.IsNil)

	// failed to close tar archive
	c.Assert(tw.Close(), checker.IsNil)

	server := fakestorage.New(c, "", fakecontext.WithBinaryFiles(map[string]*bytes.Buffer{
		"testT.tar": buffer,
	}))
	defer server.Close()

	res, b, err := request.Post("/build?remote="+server.URL()+"/testT.tar", request.ContentType("application/tar"))
	c.Assert(err, checker.IsNil)
	c.Assert(res.StatusCode, checker.Equals, http.StatusOK)
	b.Close()
}

func (s *DockerSuite) TestBuildAPIRemoteTarballContextWithCustomDockerfile(c *check.C) {
	buffer := new(bytes.Buffer)
	tw := tar.NewWriter(buffer)
	defer tw.Close()

	dockerfile := []byte(`FROM busybox
RUN echo 'wrong'`)
	err := tw.WriteHeader(&tar.Header{
		Name: "Dockerfile",
		Size: int64(len(dockerfile)),
	})
	// failed to write tar file header
	c.Assert(err, checker.IsNil)

	_, err = tw.Write(dockerfile)
	// failed to write tar file content
	c.Assert(err, checker.IsNil)

	custom := []byte(`FROM busybox
RUN echo 'right'
`)
	err = tw.WriteHeader(&tar.Header{
		Name: "custom",
		Size: int64(len(custom)),
	})

	// failed to write tar file header
	c.Assert(err, checker.IsNil)

	_, err = tw.Write(custom)
	// failed to write tar file content
	c.Assert(err, checker.IsNil)

	// failed to close tar archive
	c.Assert(tw.Close(), checker.IsNil)

	server := fakestorage.New(c, "", fakecontext.WithBinaryFiles(map[string]*bytes.Buffer{
		"testT.tar": buffer,
	}))
	defer server.Close()

	url := "/build?dockerfile=custom&remote=" + server.URL() + "/testT.tar"
	res, body, err := request.Post(url, request.ContentType("application/tar"))
	c.Assert(err, checker.IsNil)
	c.Assert(res.StatusCode, checker.Equals, http.StatusOK)

	defer body.Close()
	content, err := testutil.ReadBody(body)
	c.Assert(err, checker.IsNil)

	// Build used the wrong dockerfile.
	c.Assert(string(content), checker.Not(checker.Contains), "wrong")
}

func (s *DockerSuite) TestBuildAPILowerDockerfile(c *check.C) {
	git := fakegit.New(c, "repo", map[string]string{
		"dockerfile": `FROM busybox
RUN echo from dockerfile`,
	}, false)
	defer git.Close()

	res, body, err := request.Post("/build?remote="+git.RepoURL, request.JSON)
	c.Assert(err, checker.IsNil)
	c.Assert(res.StatusCode, checker.Equals, http.StatusOK)

	buf, err := testutil.ReadBody(body)
	c.Assert(err, checker.IsNil)

	out := string(buf)
	c.Assert(out, checker.Contains, "from dockerfile")
}

func (s *DockerSuite) TestBuildAPIBuildGitWithF(c *check.C) {
	git := fakegit.New(c, "repo", map[string]string{
		"baz": `FROM busybox
RUN echo from baz`,
		"Dockerfile": `FROM busybox
RUN echo from Dockerfile`,
	}, false)
	defer git.Close()

	// Make sure it tries to 'dockerfile' query param value
	res, body, err := request.Post("/build?dockerfile=baz&remote="+git.RepoURL, request.JSON)
	c.Assert(err, checker.IsNil)
	c.Assert(res.StatusCode, checker.Equals, http.StatusOK)

	buf, err := testutil.ReadBody(body)
	c.Assert(err, checker.IsNil)

	out := string(buf)
	c.Assert(out, checker.Contains, "from baz")
}

func (s *DockerSuite) TestBuildAPIDoubleDockerfile(c *check.C) {
	testRequires(c, UnixCli) // dockerfile overwrites Dockerfile on Windows
	git := fakegit.New(c, "repo", map[string]string{
		"Dockerfile": `FROM busybox
RUN echo from Dockerfile`,
		"dockerfile": `FROM busybox
RUN echo from dockerfile`,
	}, false)
	defer git.Close()

	// Make sure it tries to 'dockerfile' query param value
	res, body, err := request.Post("/build?remote="+git.RepoURL, request.JSON)
	c.Assert(err, checker.IsNil)
	c.Assert(res.StatusCode, checker.Equals, http.StatusOK)

	buf, err := testutil.ReadBody(body)
	c.Assert(err, checker.IsNil)

	out := string(buf)
	c.Assert(out, checker.Contains, "from Dockerfile")
}

func (s *DockerSuite) TestBuildAPIUnnormalizedTarPaths(c *check.C) {
	// Make sure that build context tars with entries of the form
	// x/./y don't cause caching false positives.

	buildFromTarContext := func(fileContents []byte) string {
		buffer := new(bytes.Buffer)
		tw := tar.NewWriter(buffer)
		defer tw.Close()

		dockerfile := []byte(`FROM busybox
	COPY dir /dir/`)
		err := tw.WriteHeader(&tar.Header{
			Name: "Dockerfile",
			Size: int64(len(dockerfile)),
		})
		//failed to write tar file header
		c.Assert(err, checker.IsNil)

		_, err = tw.Write(dockerfile)
		// failed to write Dockerfile in tar file content
		c.Assert(err, checker.IsNil)

		err = tw.WriteHeader(&tar.Header{
			Name: "dir/./file",
			Size: int64(len(fileContents)),
		})
		//failed to write tar file header
		c.Assert(err, checker.IsNil)

		_, err = tw.Write(fileContents)
		// failed to write file contents in tar file content
		c.Assert(err, checker.IsNil)

		// failed to close tar archive
		c.Assert(tw.Close(), checker.IsNil)

		res, body, err := request.Post("/build", request.RawContent(ioutil.NopCloser(buffer)), request.ContentType("application/x-tar"))
		c.Assert(err, checker.IsNil)
		c.Assert(res.StatusCode, checker.Equals, http.StatusOK)

		out, err := testutil.ReadBody(body)
		c.Assert(err, checker.IsNil)
		lines := strings.Split(string(out), "\n")
		c.Assert(len(lines), checker.GreaterThan, 1)
		c.Assert(lines[len(lines)-2], checker.Matches, ".*Successfully built [0-9a-f]{12}.*")

		re := regexp.MustCompile("Successfully built ([0-9a-f]{12})")
		matches := re.FindStringSubmatch(lines[len(lines)-2])
		return matches[1]
	}

	imageA := buildFromTarContext([]byte("abc"))
	imageB := buildFromTarContext([]byte("def"))

	c.Assert(imageA, checker.Not(checker.Equals), imageB)
}

func (s *DockerSuite) TestBuildOnBuildWithCopy(c *check.C) {
	dockerfile := `
		FROM ` + minimalBaseImage() + ` as onbuildbase
		ONBUILD COPY file /file

		FROM onbuildbase
	`
	ctx := fakecontext.New(c, "",
		fakecontext.WithDockerfile(dockerfile),
		fakecontext.WithFile("file", "some content"),
	)
	defer ctx.Close()

	res, body, err := request.Post(
		"/build",
		request.RawContent(ctx.AsTarReader(c)),
		request.ContentType("application/x-tar"))
	c.Assert(err, checker.IsNil)
	c.Assert(res.StatusCode, checker.Equals, http.StatusOK)

	out, err := testutil.ReadBody(body)
	c.Assert(err, checker.IsNil)
	c.Assert(string(out), checker.Contains, "Successfully built")
}

func (s *DockerSuite) TestBuildOnBuildCache(c *check.C) {
	build := func(dockerfile string) []byte {
		ctx := fakecontext.New(c, "",
			fakecontext.WithDockerfile(dockerfile),
		)
		defer ctx.Close()

		res, body, err := request.Post(
			"/build",
			request.RawContent(ctx.AsTarReader(c)),
			request.ContentType("application/x-tar"))
		require.NoError(c, err)
		assert.Equal(c, http.StatusOK, res.StatusCode)

		out, err := testutil.ReadBody(body)
		require.NoError(c, err)
		assert.Contains(c, string(out), "Successfully built")
		return out
	}

	dockerfile := `
		FROM ` + minimalBaseImage() + ` as onbuildbase
		ENV something=bar
		ONBUILD ENV foo=bar
	`
	build(dockerfile)

	dockerfile += "FROM onbuildbase"
	out := build(dockerfile)

	imageIDs := getImageIDsFromBuild(c, out)
	assert.Len(c, imageIDs, 2)
	parentID, childID := imageIDs[0], imageIDs[1]

	client, err := request.NewClient()
	require.NoError(c, err)

	// check parentID is correct
	image, _, err := client.ImageInspectWithRaw(context.Background(), childID)
	require.NoError(c, err)
	assert.Equal(c, parentID, image.Parent)
}

func (s *DockerRegistrySuite) TestBuildCopyFromForcePull(c *check.C) {
	client, err := request.NewClient()
	require.NoError(c, err)

	repoName := fmt.Sprintf("%v/dockercli/busybox", privateRegistryURL)
	// tag the image to upload it to the private registry
	err = client.ImageTag(context.TODO(), "busybox", repoName)
	assert.Nil(c, err)
	// push the image to the registry
	rc, err := client.ImagePush(context.TODO(), repoName, types.ImagePushOptions{RegistryAuth: "{}"})
	assert.Nil(c, err)
	_, err = io.Copy(ioutil.Discard, rc)
	assert.Nil(c, err)

	dockerfile := fmt.Sprintf(`
		FROM %s AS foo
		RUN touch abc
		FROM %s
		COPY --from=foo /abc /
		`, repoName, repoName)

	ctx := fakecontext.New(c, "",
		fakecontext.WithDockerfile(dockerfile),
	)
	defer ctx.Close()

	res, body, err := request.Post(
		"/build?pull=1",
		request.RawContent(ctx.AsTarReader(c)),
		request.ContentType("application/x-tar"))
	require.NoError(c, err)
	assert.Equal(c, http.StatusOK, res.StatusCode)

	out, err := testutil.ReadBody(body)
	require.NoError(c, err)
	assert.Contains(c, string(out), "Successfully built")
}

func (s *DockerSuite) TestBuildAddRemoteNoDecompress(c *check.C) {
	buffer := new(bytes.Buffer)
	tw := tar.NewWriter(buffer)
	dt := []byte("contents")
	err := tw.WriteHeader(&tar.Header{
		Name:     "foo",
		Size:     int64(len(dt)),
		Mode:     0600,
		Typeflag: tar.TypeReg,
	})
	require.NoError(c, err)
	_, err = tw.Write(dt)
	require.NoError(c, err)
	err = tw.Close()
	require.NoError(c, err)

	server := fakestorage.New(c, "", fakecontext.WithBinaryFiles(map[string]*bytes.Buffer{
		"test.tar": buffer,
	}))
	defer server.Close()

	dockerfile := fmt.Sprintf(`
		FROM busybox
		ADD %s/test.tar /
		RUN [ -f test.tar ]
		`, server.URL())

	ctx := fakecontext.New(c, "",
		fakecontext.WithDockerfile(dockerfile),
	)
	defer ctx.Close()

	res, body, err := request.Post(
		"/build",
		request.RawContent(ctx.AsTarReader(c)),
		request.ContentType("application/x-tar"))
	require.NoError(c, err)
	assert.Equal(c, http.StatusOK, res.StatusCode)

	out, err := testutil.ReadBody(body)
	require.NoError(c, err)
	assert.Contains(c, string(out), "Successfully built")
}

func (s *DockerSuite) TestBuildWithSession(c *check.C) {
	testRequires(c, ExperimentalDaemon)

	dockerfile := `
		FROM busybox
		COPY file /
		RUN cat /file
	`

	fctx := fakecontext.New(c, "",
		fakecontext.WithFile("file", "some content"),
	)
	defer fctx.Close()

	out := testBuildWithSession(c, fctx.Dir, dockerfile)
	assert.Contains(c, out, "some content")

	fctx.Add("second", "contentcontent")

	dockerfile += `
	COPY second /
	RUN cat /second
	`

	out = testBuildWithSession(c, fctx.Dir, dockerfile)
	assert.Equal(c, strings.Count(out, "Using cache"), 2)
	assert.Contains(c, out, "contentcontent")

	client, err := request.NewClient()
	require.NoError(c, err)

	du, err := client.DiskUsage(context.TODO())
	assert.Nil(c, err)
	assert.True(c, du.BuilderSize > 10)

	out = testBuildWithSession(c, fctx.Dir, dockerfile)
	assert.Equal(c, strings.Count(out, "Using cache"), 4)

	du2, err := client.DiskUsage(context.TODO())
	assert.Nil(c, err)
	assert.Equal(c, du.BuilderSize, du2.BuilderSize)

	// rebuild with regular tar, confirm cache still applies
	fctx.Add("Dockerfile", dockerfile)
	res, body, err := request.Post(
		"/build",
		request.RawContent(fctx.AsTarReader(c)),
		request.ContentType("application/x-tar"))
	require.NoError(c, err)
	assert.Equal(c, http.StatusOK, res.StatusCode)

	outBytes, err := testutil.ReadBody(body)
	require.NoError(c, err)
	assert.Contains(c, string(outBytes), "Successfully built")
	assert.Equal(c, strings.Count(string(outBytes), "Using cache"), 4)

	_, err = client.BuildCachePrune(context.TODO())
	assert.Nil(c, err)

	du, err = client.DiskUsage(context.TODO())
	assert.Nil(c, err)
	assert.Equal(c, du.BuilderSize, int64(0))
}

func testBuildWithSession(c *check.C, dir, dockerfile string) (outStr string) {
	client, err := request.NewClient()
	require.NoError(c, err)

	sess, err := session.NewSession("foo1", "foo")
	assert.Nil(c, err)

	fsProvider := filesync.NewFSSyncProvider(dir, nil)
	sess.Allow(fsProvider)

	g, ctx := errgroup.WithContext(context.Background())

	g.Go(func() error {
		return sess.Run(ctx, client.DialSession)
	})

	g.Go(func() error {
		res, body, err := request.Post("/build?remote=client-session&session="+sess.UUID(), func(req *http.Request) error {
			req.Body = ioutil.NopCloser(strings.NewReader(dockerfile))
			return nil
		})
		if err != nil {
			return err
		}
		assert.Equal(c, res.StatusCode, http.StatusOK)
		out, err := testutil.ReadBody(body)
		require.NoError(c, err)
		assert.Contains(c, string(out), "Successfully built")
		sess.Close()
		outStr = string(out)
		return nil
	})

	err = g.Wait()
	assert.Nil(c, err)
	return
}

type buildLine struct {
	Stream string
	Aux    struct {
		ID string
	}
}

func getImageIDsFromBuild(c *check.C, output []byte) []string {
	ids := []string{}
	for _, line := range bytes.Split(output, []byte("\n")) {
		if len(line) == 0 {
			continue
		}
		entry := buildLine{}
		require.NoError(c, json.Unmarshal(line, &entry))
		if entry.Aux.ID != "" {
			ids = append(ids, entry.Aux.ID)
		}
	}
	return ids
}
