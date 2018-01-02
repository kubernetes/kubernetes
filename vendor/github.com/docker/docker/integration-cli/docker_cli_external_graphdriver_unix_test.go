// +build !windows

package main

import (
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"

	"github.com/docker/docker/daemon/graphdriver"
	"github.com/docker/docker/daemon/graphdriver/vfs"
	"github.com/docker/docker/integration-cli/daemon"
	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/plugins"
	"github.com/go-check/check"
)

func init() {
	check.Suite(&DockerExternalGraphdriverSuite{
		ds: &DockerSuite{},
	})
}

type DockerExternalGraphdriverSuite struct {
	server  *httptest.Server
	jserver *httptest.Server
	ds      *DockerSuite
	d       *daemon.Daemon
	ec      map[string]*graphEventsCounter
}

type graphEventsCounter struct {
	activations int
	creations   int
	removals    int
	gets        int
	puts        int
	stats       int
	cleanups    int
	exists      int
	init        int
	metadata    int
	diff        int
	applydiff   int
	changes     int
	diffsize    int
}

func (s *DockerExternalGraphdriverSuite) SetUpTest(c *check.C) {
	s.d = daemon.New(c, dockerBinary, dockerdBinary, daemon.Config{
		Experimental: testEnv.ExperimentalDaemon(),
	})
}

func (s *DockerExternalGraphdriverSuite) OnTimeout(c *check.C) {
	s.d.DumpStackAndQuit()
}

func (s *DockerExternalGraphdriverSuite) TearDownTest(c *check.C) {
	if s.d != nil {
		s.d.Stop(c)
		s.ds.TearDownTest(c)
	}
}

func (s *DockerExternalGraphdriverSuite) SetUpSuite(c *check.C) {
	s.ec = make(map[string]*graphEventsCounter)
	s.setUpPluginViaSpecFile(c)
	s.setUpPluginViaJSONFile(c)
}

func (s *DockerExternalGraphdriverSuite) setUpPluginViaSpecFile(c *check.C) {
	mux := http.NewServeMux()
	s.server = httptest.NewServer(mux)

	s.setUpPlugin(c, "test-external-graph-driver", "spec", mux, []byte(s.server.URL))
}

func (s *DockerExternalGraphdriverSuite) setUpPluginViaJSONFile(c *check.C) {
	mux := http.NewServeMux()
	s.jserver = httptest.NewServer(mux)

	p := plugins.NewLocalPlugin("json-external-graph-driver", s.jserver.URL)
	b, err := json.Marshal(p)
	c.Assert(err, check.IsNil)

	s.setUpPlugin(c, "json-external-graph-driver", "json", mux, b)
}

func (s *DockerExternalGraphdriverSuite) setUpPlugin(c *check.C, name string, ext string, mux *http.ServeMux, b []byte) {
	type graphDriverRequest struct {
		ID         string `json:",omitempty"`
		Parent     string `json:",omitempty"`
		MountLabel string `json:",omitempty"`
		ReadOnly   bool   `json:",omitempty"`
	}

	type graphDriverResponse struct {
		Err      error             `json:",omitempty"`
		Dir      string            `json:",omitempty"`
		Exists   bool              `json:",omitempty"`
		Status   [][2]string       `json:",omitempty"`
		Metadata map[string]string `json:",omitempty"`
		Changes  []archive.Change  `json:",omitempty"`
		Size     int64             `json:",omitempty"`
	}

	respond := func(w http.ResponseWriter, data interface{}) {
		w.Header().Set("Content-Type", "application/vnd.docker.plugins.v1+json")
		switch t := data.(type) {
		case error:
			fmt.Fprintln(w, fmt.Sprintf(`{"Err": %q}`, t.Error()))
		case string:
			fmt.Fprintln(w, t)
		default:
			json.NewEncoder(w).Encode(&data)
		}
	}

	decReq := func(b io.ReadCloser, out interface{}, w http.ResponseWriter) error {
		defer b.Close()
		if err := json.NewDecoder(b).Decode(&out); err != nil {
			http.Error(w, fmt.Sprintf("error decoding json: %s", err.Error()), 500)
		}
		return nil
	}

	base, err := ioutil.TempDir("", name)
	c.Assert(err, check.IsNil)
	vfsProto, err := vfs.Init(base, []string{}, nil, nil)
	c.Assert(err, check.IsNil, check.Commentf("error initializing graph driver"))
	driver := graphdriver.NewNaiveDiffDriver(vfsProto, nil, nil)

	s.ec[ext] = &graphEventsCounter{}
	mux.HandleFunc("/Plugin.Activate", func(w http.ResponseWriter, r *http.Request) {
		s.ec[ext].activations++
		respond(w, `{"Implements": ["GraphDriver"]}`)
	})

	mux.HandleFunc("/GraphDriver.Init", func(w http.ResponseWriter, r *http.Request) {
		s.ec[ext].init++
		respond(w, "{}")
	})

	mux.HandleFunc("/GraphDriver.CreateReadWrite", func(w http.ResponseWriter, r *http.Request) {
		s.ec[ext].creations++

		var req graphDriverRequest
		if err := decReq(r.Body, &req, w); err != nil {
			return
		}
		if err := driver.CreateReadWrite(req.ID, req.Parent, nil); err != nil {
			respond(w, err)
			return
		}
		respond(w, "{}")
	})

	mux.HandleFunc("/GraphDriver.Create", func(w http.ResponseWriter, r *http.Request) {
		s.ec[ext].creations++

		var req graphDriverRequest
		if err := decReq(r.Body, &req, w); err != nil {
			return
		}
		if err := driver.Create(req.ID, req.Parent, nil); err != nil {
			respond(w, err)
			return
		}
		respond(w, "{}")
	})

	mux.HandleFunc("/GraphDriver.Remove", func(w http.ResponseWriter, r *http.Request) {
		s.ec[ext].removals++

		var req graphDriverRequest
		if err := decReq(r.Body, &req, w); err != nil {
			return
		}

		if err := driver.Remove(req.ID); err != nil {
			respond(w, err)
			return
		}
		respond(w, "{}")
	})

	mux.HandleFunc("/GraphDriver.Get", func(w http.ResponseWriter, r *http.Request) {
		s.ec[ext].gets++

		var req graphDriverRequest
		if err := decReq(r.Body, &req, w); err != nil {
			return
		}

		dir, err := driver.Get(req.ID, req.MountLabel)
		if err != nil {
			respond(w, err)
			return
		}
		respond(w, &graphDriverResponse{Dir: dir})
	})

	mux.HandleFunc("/GraphDriver.Put", func(w http.ResponseWriter, r *http.Request) {
		s.ec[ext].puts++

		var req graphDriverRequest
		if err := decReq(r.Body, &req, w); err != nil {
			return
		}

		if err := driver.Put(req.ID); err != nil {
			respond(w, err)
			return
		}
		respond(w, "{}")
	})

	mux.HandleFunc("/GraphDriver.Exists", func(w http.ResponseWriter, r *http.Request) {
		s.ec[ext].exists++

		var req graphDriverRequest
		if err := decReq(r.Body, &req, w); err != nil {
			return
		}
		respond(w, &graphDriverResponse{Exists: driver.Exists(req.ID)})
	})

	mux.HandleFunc("/GraphDriver.Status", func(w http.ResponseWriter, r *http.Request) {
		s.ec[ext].stats++
		respond(w, &graphDriverResponse{Status: driver.Status()})
	})

	mux.HandleFunc("/GraphDriver.Cleanup", func(w http.ResponseWriter, r *http.Request) {
		s.ec[ext].cleanups++
		err := driver.Cleanup()
		if err != nil {
			respond(w, err)
			return
		}
		respond(w, `{}`)
	})

	mux.HandleFunc("/GraphDriver.GetMetadata", func(w http.ResponseWriter, r *http.Request) {
		s.ec[ext].metadata++

		var req graphDriverRequest
		if err := decReq(r.Body, &req, w); err != nil {
			return
		}

		data, err := driver.GetMetadata(req.ID)
		if err != nil {
			respond(w, err)
			return
		}
		respond(w, &graphDriverResponse{Metadata: data})
	})

	mux.HandleFunc("/GraphDriver.Diff", func(w http.ResponseWriter, r *http.Request) {
		s.ec[ext].diff++

		var req graphDriverRequest
		if err := decReq(r.Body, &req, w); err != nil {
			return
		}

		diff, err := driver.Diff(req.ID, req.Parent)
		if err != nil {
			respond(w, err)
			return
		}
		io.Copy(w, diff)
	})

	mux.HandleFunc("/GraphDriver.Changes", func(w http.ResponseWriter, r *http.Request) {
		s.ec[ext].changes++
		var req graphDriverRequest
		if err := decReq(r.Body, &req, w); err != nil {
			return
		}

		changes, err := driver.Changes(req.ID, req.Parent)
		if err != nil {
			respond(w, err)
			return
		}
		respond(w, &graphDriverResponse{Changes: changes})
	})

	mux.HandleFunc("/GraphDriver.ApplyDiff", func(w http.ResponseWriter, r *http.Request) {
		s.ec[ext].applydiff++
		diff := r.Body
		defer r.Body.Close()

		id := r.URL.Query().Get("id")
		parent := r.URL.Query().Get("parent")

		if id == "" {
			http.Error(w, fmt.Sprintf("missing id"), 409)
		}

		size, err := driver.ApplyDiff(id, parent, diff)
		if err != nil {
			respond(w, err)
			return
		}
		respond(w, &graphDriverResponse{Size: size})
	})

	mux.HandleFunc("/GraphDriver.DiffSize", func(w http.ResponseWriter, r *http.Request) {
		s.ec[ext].diffsize++

		var req graphDriverRequest
		if err := decReq(r.Body, &req, w); err != nil {
			return
		}

		size, err := driver.DiffSize(req.ID, req.Parent)
		if err != nil {
			respond(w, err)
			return
		}
		respond(w, &graphDriverResponse{Size: size})
	})

	err = os.MkdirAll("/etc/docker/plugins", 0755)
	c.Assert(err, check.IsNil, check.Commentf("error creating /etc/docker/plugins"))

	specFile := "/etc/docker/plugins/" + name + "." + ext
	err = ioutil.WriteFile(specFile, b, 0644)
	c.Assert(err, check.IsNil, check.Commentf("error writing to %s", specFile))
}

func (s *DockerExternalGraphdriverSuite) TearDownSuite(c *check.C) {
	s.server.Close()
	s.jserver.Close()

	err := os.RemoveAll("/etc/docker/plugins")
	c.Assert(err, check.IsNil, check.Commentf("error removing /etc/docker/plugins"))
}

func (s *DockerExternalGraphdriverSuite) TestExternalGraphDriver(c *check.C) {
	testRequires(c, ExperimentalDaemon)

	s.testExternalGraphDriver("test-external-graph-driver", "spec", c)
	s.testExternalGraphDriver("json-external-graph-driver", "json", c)
}

func (s *DockerExternalGraphdriverSuite) testExternalGraphDriver(name string, ext string, c *check.C) {
	s.d.StartWithBusybox(c, "-s", name)

	out, err := s.d.Cmd("run", "--name=graphtest", "busybox", "sh", "-c", "echo hello > /hello")
	c.Assert(err, check.IsNil, check.Commentf(out))

	s.d.Restart(c, "-s", name)

	out, err = s.d.Cmd("inspect", "--format={{.GraphDriver.Name}}", "graphtest")
	c.Assert(err, check.IsNil, check.Commentf(out))
	c.Assert(strings.TrimSpace(out), check.Equals, name)

	out, err = s.d.Cmd("diff", "graphtest")
	c.Assert(err, check.IsNil, check.Commentf(out))
	c.Assert(strings.Contains(out, "A /hello"), check.Equals, true, check.Commentf("diff output: %s", out))

	out, err = s.d.Cmd("rm", "-f", "graphtest")
	c.Assert(err, check.IsNil, check.Commentf(out))

	out, err = s.d.Cmd("info")
	c.Assert(err, check.IsNil, check.Commentf(out))

	s.d.Stop(c)

	// Don't check s.ec.exists, because the daemon no longer calls the
	// Exists function.
	c.Assert(s.ec[ext].activations, check.Equals, 2)
	c.Assert(s.ec[ext].init, check.Equals, 2)
	c.Assert(s.ec[ext].creations >= 1, check.Equals, true)
	c.Assert(s.ec[ext].removals >= 1, check.Equals, true)
	c.Assert(s.ec[ext].gets >= 1, check.Equals, true)
	c.Assert(s.ec[ext].puts >= 1, check.Equals, true)
	c.Assert(s.ec[ext].stats, check.Equals, 5)
	c.Assert(s.ec[ext].cleanups, check.Equals, 2)
	c.Assert(s.ec[ext].applydiff >= 1, check.Equals, true)
	c.Assert(s.ec[ext].changes, check.Equals, 1)
	c.Assert(s.ec[ext].diffsize, check.Equals, 0)
	c.Assert(s.ec[ext].diff, check.Equals, 0)
	c.Assert(s.ec[ext].metadata, check.Equals, 1)
}

func (s *DockerExternalGraphdriverSuite) TestExternalGraphDriverPull(c *check.C) {
	testRequires(c, Network, ExperimentalDaemon)

	s.d.Start(c)

	out, err := s.d.Cmd("pull", "busybox:latest")
	c.Assert(err, check.IsNil, check.Commentf(out))

	out, err = s.d.Cmd("run", "-d", "busybox", "top")
	c.Assert(err, check.IsNil, check.Commentf(out))
}
