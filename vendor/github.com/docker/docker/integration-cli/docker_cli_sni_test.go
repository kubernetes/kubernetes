package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os/exec"
	"strings"

	"github.com/go-check/check"
)

func (s *DockerSuite) TestClientSetsTLSServerName(c *check.C) {
	c.Skip("Flakey test")
	// there may be more than one hit to the server for each registry request
	serverNameReceived := []string{}
	var serverName string

	virtualHostServer := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		serverNameReceived = append(serverNameReceived, r.TLS.ServerName)
	}))
	defer virtualHostServer.Close()
	// discard TLS handshake errors written by default to os.Stderr
	virtualHostServer.Config.ErrorLog = log.New(ioutil.Discard, "", 0)

	u, err := url.Parse(virtualHostServer.URL)
	c.Assert(err, check.IsNil)
	hostPort := u.Host
	serverName = strings.Split(hostPort, ":")[0]

	repoName := fmt.Sprintf("%v/dockercli/image:latest", hostPort)
	cmd := exec.Command(dockerBinary, "pull", repoName)
	cmd.Run()

	// check that the fake server was hit at least once
	c.Assert(len(serverNameReceived) > 0, check.Equals, true)
	// check that for each hit the right server name was received
	for _, item := range serverNameReceived {
		c.Check(item, check.Equals, serverName)
	}
}
