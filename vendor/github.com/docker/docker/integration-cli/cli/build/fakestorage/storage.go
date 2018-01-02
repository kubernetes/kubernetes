package fakestorage

import (
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"sync"

	"github.com/docker/docker/integration-cli/cli"
	"github.com/docker/docker/integration-cli/cli/build"
	"github.com/docker/docker/integration-cli/cli/build/fakecontext"
	"github.com/docker/docker/integration-cli/environment"
	"github.com/docker/docker/integration-cli/request"
	"github.com/docker/docker/pkg/stringutils"
)

var (
	testEnv  *environment.Execution
	onlyOnce sync.Once
)

// EnsureTestEnvIsLoaded make sure the test environment is loaded for this package
func EnsureTestEnvIsLoaded(t testingT) {
	var doIt bool
	var err error
	onlyOnce.Do(func() {
		doIt = true
	})

	if !doIt {
		return
	}
	testEnv, err = environment.New()
	if err != nil {
		t.Fatalf("error loading testenv : %v", err)
	}
}

type testingT interface {
	logT
	Fatal(args ...interface{})
	Fatalf(string, ...interface{})
}

type logT interface {
	Logf(string, ...interface{})
}

// Fake is a static file server. It might be running locally or remotely
// on test host.
type Fake interface {
	Close() error
	URL() string
	CtxDir() string
}

// New returns a static file server that will be use as build context.
func New(t testingT, dir string, modifiers ...func(*fakecontext.Fake) error) Fake {
	ctx := fakecontext.New(t, dir, modifiers...)
	if testEnv.LocalDaemon() {
		return newLocalFakeStorage(t, ctx)
	}
	return newRemoteFileServer(t, ctx)
}

// localFileStorage is a file storage on the running machine
type localFileStorage struct {
	*fakecontext.Fake
	*httptest.Server
}

func (s *localFileStorage) URL() string {
	return s.Server.URL
}

func (s *localFileStorage) CtxDir() string {
	return s.Fake.Dir
}

func (s *localFileStorage) Close() error {
	defer s.Server.Close()
	return s.Fake.Close()
}

func newLocalFakeStorage(t testingT, ctx *fakecontext.Fake) *localFileStorage {
	handler := http.FileServer(http.Dir(ctx.Dir))
	server := httptest.NewServer(handler)
	return &localFileStorage{
		Fake:   ctx,
		Server: server,
	}
}

// remoteFileServer is a containerized static file server started on the remote
// testing machine to be used in URL-accepting docker build functionality.
type remoteFileServer struct {
	host      string // hostname/port web server is listening to on docker host e.g. 0.0.0.0:43712
	container string
	image     string
	ctx       *fakecontext.Fake
}

func (f *remoteFileServer) URL() string {
	u := url.URL{
		Scheme: "http",
		Host:   f.host}
	return u.String()
}

func (f *remoteFileServer) CtxDir() string {
	return f.ctx.Dir
}

func (f *remoteFileServer) Close() error {
	defer func() {
		if f.ctx != nil {
			f.ctx.Close()
		}
		if f.image != "" {
			if err := cli.Docker(cli.Args("rmi", "-f", f.image)).Error; err != nil {
				fmt.Fprintf(os.Stderr, "Error closing remote file server : %v\n", err)
			}
		}
	}()
	if f.container == "" {
		return nil
	}
	return cli.Docker(cli.Args("rm", "-fv", f.container)).Error
}

func newRemoteFileServer(t testingT, ctx *fakecontext.Fake) *remoteFileServer {
	var (
		image     = fmt.Sprintf("fileserver-img-%s", strings.ToLower(stringutils.GenerateRandomAlphaOnlyString(10)))
		container = fmt.Sprintf("fileserver-cnt-%s", strings.ToLower(stringutils.GenerateRandomAlphaOnlyString(10)))
	)

	ensureHTTPServerImage(t)

	// Build the image
	if err := ctx.Add("Dockerfile", `FROM httpserver
COPY . /static`); err != nil {
		t.Fatal(err)
	}
	cli.BuildCmd(t, image, build.WithoutCache, build.WithExternalBuildContext(ctx))

	// Start the container
	cli.DockerCmd(t, "run", "-d", "-P", "--name", container, image)

	// Find out the system assigned port
	out := cli.DockerCmd(t, "port", container, "80/tcp").Combined()
	fileserverHostPort := strings.Trim(out, "\n")
	_, port, err := net.SplitHostPort(fileserverHostPort)
	if err != nil {
		t.Fatalf("unable to parse file server host:port: %v", err)
	}

	dockerHostURL, err := url.Parse(request.DaemonHost())
	if err != nil {
		t.Fatalf("unable to parse daemon host URL: %v", err)
	}

	host, _, err := net.SplitHostPort(dockerHostURL.Host)
	if err != nil {
		t.Fatalf("unable to parse docker daemon host:port: %v", err)
	}

	return &remoteFileServer{
		container: container,
		image:     image,
		host:      fmt.Sprintf("%s:%s", host, port),
		ctx:       ctx}
}
