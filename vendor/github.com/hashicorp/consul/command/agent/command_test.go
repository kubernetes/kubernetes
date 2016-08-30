package agent

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/hashicorp/consul/testutil"
	"github.com/mitchellh/cli"
)

func TestCommand_implements(t *testing.T) {
	var _ cli.Command = new(Command)
}

func TestValidDatacenter(t *testing.T) {
	shouldMatch := []string{
		"dc1",
		"east-aws-001",
		"PROD_aws01-small",
	}
	noMatch := []string{
		"east.aws",
		"east!aws",
		"first,second",
	}
	for _, m := range shouldMatch {
		if !validDatacenter.MatchString(m) {
			t.Fatalf("expected match: %s", m)
		}
	}
	for _, m := range noMatch {
		if validDatacenter.MatchString(m) {
			t.Fatalf("expected no match: %s", m)
		}
	}
}

func TestRetryJoin(t *testing.T) {
	dir, agent := makeAgent(t, nextConfig())
	defer os.RemoveAll(dir)
	defer agent.Shutdown()

	conf2 := nextConfig()
	tmpDir, err := ioutil.TempDir("", "consul")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	defer os.RemoveAll(tmpDir)

	doneCh := make(chan struct{})
	shutdownCh := make(chan struct{})

	defer func() {
		close(shutdownCh)
		<-doneCh
	}()

	cmd := &Command{
		ShutdownCh: shutdownCh,
		Ui:         new(cli.MockUi),
	}

	serfAddr := fmt.Sprintf(
		"%s:%d",
		agent.config.BindAddr,
		agent.config.Ports.SerfLan)

	serfWanAddr := fmt.Sprintf(
		"%s:%d",
		agent.config.BindAddr,
		agent.config.Ports.SerfWan)

	args := []string{
		"-server",
		"-data-dir", tmpDir,
		"-node", fmt.Sprintf(`"%s"`, conf2.NodeName),
		"-advertise", agent.config.BindAddr,
		"-retry-join", serfAddr,
		"-retry-interval", "1s",
		"-retry-join-wan", serfWanAddr,
		"-retry-interval-wan", "1s",
	}

	go func() {
		if code := cmd.Run(args); code != 0 {
			log.Printf("bad: %d", code)
		}
		close(doneCh)
	}()

	testutil.WaitForResult(func() (bool, error) {
		mem := agent.LANMembers()
		if len(mem) != 2 {
			return false, fmt.Errorf("bad: %#v", mem)
		}
		mem = agent.WANMembers()
		if len(mem) != 2 {
			return false, fmt.Errorf("bad (wan): %#v", mem)
		}
		return true, nil
	}, func(err error) {
		t.Fatalf(err.Error())
	})
}

func TestReadCliConfig(t *testing.T) {

	shutdownCh := make(chan struct{})
	defer close(shutdownCh)

	tmpDir, err := ioutil.TempDir("", "consul")
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	cmd := &Command{
		args: []string{
			"-data-dir", tmpDir,
			"-node", `"a"`,
			"-advertise-wan", "1.2.3.4",
		},
		ShutdownCh: shutdownCh,
		Ui:         new(cli.MockUi),
	}

	config := cmd.readConfig()
	if config.AdvertiseAddrWan != "1.2.3.4" {
		t.Fatalf("expected -advertise-addr-wan 1.2.3.4 got %s", config.AdvertiseAddrWan)
	}
}

func TestRetryJoinFail(t *testing.T) {
	conf := nextConfig()
	tmpDir, err := ioutil.TempDir("", "consul")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	defer os.RemoveAll(tmpDir)

	shutdownCh := make(chan struct{})
	defer close(shutdownCh)

	cmd := &Command{
		ShutdownCh: shutdownCh,
		Ui:         new(cli.MockUi),
	}

	serfAddr := fmt.Sprintf("%s:%d", conf.BindAddr, conf.Ports.SerfLan)

	args := []string{
		"-data-dir", tmpDir,
		"-retry-join", serfAddr,
		"-retry-max", "1",
	}

	if code := cmd.Run(args); code == 0 {
		t.Fatalf("bad: %d", code)
	}
}

func TestRetryJoinWanFail(t *testing.T) {
	conf := nextConfig()
	tmpDir, err := ioutil.TempDir("", "consul")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	defer os.RemoveAll(tmpDir)

	shutdownCh := make(chan struct{})
	defer close(shutdownCh)

	cmd := &Command{
		ShutdownCh: shutdownCh,
		Ui:         new(cli.MockUi),
	}

	serfAddr := fmt.Sprintf("%s:%d", conf.BindAddr, conf.Ports.SerfWan)

	args := []string{
		"-server",
		"-data-dir", tmpDir,
		"-retry-join-wan", serfAddr,
		"-retry-max-wan", "1",
	}

	if code := cmd.Run(args); code == 0 {
		t.Fatalf("bad: %d", code)
	}
}

func TestSetupAgent_RPCUnixSocket_FileExists(t *testing.T) {
	conf := nextConfig()
	tmpDir, err := ioutil.TempDir("", "consul")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	defer os.RemoveAll(tmpDir)

	tmpFile, err := ioutil.TempFile("", "consul")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	defer os.Remove(tmpFile.Name())
	socketPath := tmpFile.Name()

	conf.DataDir = tmpDir
	conf.Server = true
	conf.Bootstrap = true

	// Set socket address to an existing file.
	conf.Addresses.RPC = "unix://" + socketPath

	// Custom mode for socket file
	conf.UnixSockets.Perms = "0777"

	shutdownCh := make(chan struct{})
	defer close(shutdownCh)

	cmd := &Command{
		ShutdownCh: shutdownCh,
		Ui:         new(cli.MockUi),
	}

	logWriter := NewLogWriter(512)
	logOutput := new(bytes.Buffer)

	// Ensure the server is created
	if err := cmd.setupAgent(conf, logOutput, logWriter); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Ensure the file was replaced by the socket
	fi, err := os.Stat(socketPath)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if fi.Mode()&os.ModeSocket == 0 {
		t.Fatalf("expected socket to replace file")
	}

	// Ensure permissions were applied to the socket file
	if fi.Mode().String() != "Srwxrwxrwx" {
		t.Fatalf("bad permissions: %s", fi.Mode())
	}
}

func TestSetupScadaConn(t *testing.T) {
	// Create a config and assign an infra name
	conf1 := nextConfig()
	conf1.AtlasInfrastructure = "hashicorp/test1"
	conf1.AtlasToken = "abc"

	dir, agent := makeAgent(t, conf1)
	defer os.RemoveAll(dir)
	defer agent.Shutdown()

	cmd := &Command{
		ShutdownCh: make(chan struct{}),
		Ui:         new(cli.MockUi),
		agent:      agent,
	}

	// First start creates the scada conn
	if err := cmd.setupScadaConn(conf1); err != nil {
		t.Fatalf("err: %s", err)
	}
	list := cmd.scadaHttp.listener.(*scadaListener)
	if list == nil || list.addr.infra != "hashicorp/test1" {
		t.Fatalf("bad: %#v", list)
	}
	http1 := cmd.scadaHttp
	provider1 := cmd.scadaProvider

	// Performing setup again tears down original and replaces
	// with a new SCADA client.
	conf2 := nextConfig()
	conf2.AtlasInfrastructure = "hashicorp/test2"
	conf2.AtlasToken = "123"
	if err := cmd.setupScadaConn(conf2); err != nil {
		t.Fatalf("err: %s", err)
	}
	if cmd.scadaHttp == http1 || cmd.scadaProvider == provider1 {
		t.Fatalf("should change: %#v %#v", cmd.scadaHttp, cmd.scadaProvider)
	}
	list = cmd.scadaHttp.listener.(*scadaListener)
	if list == nil || list.addr.infra != "hashicorp/test2" {
		t.Fatalf("bad: %#v", list)
	}

	// Original provider and listener must be closed
	if !provider1.IsShutdown() {
		t.Fatalf("should be shutdown")
	}
	if _, err := http1.listener.Accept(); !strings.Contains(err.Error(), "closed") {
		t.Fatalf("should be closed")
	}
}

func TestProtectDataDir(t *testing.T) {
	dir, err := ioutil.TempDir("", "consul")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	defer os.RemoveAll(dir)

	if err := os.MkdirAll(filepath.Join(dir, "mdb"), 0700); err != nil {
		t.Fatalf("err: %v", err)
	}

	cfgFile, err := ioutil.TempFile("", "consul")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	defer os.Remove(cfgFile.Name())

	content := fmt.Sprintf(`{"server": true, "data_dir": "%s"}`, dir)
	_, err = cfgFile.Write([]byte(content))
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	ui := new(cli.MockUi)
	cmd := &Command{
		Ui:   ui,
		args: []string{"-config-file=" + cfgFile.Name()},
	}
	if conf := cmd.readConfig(); conf != nil {
		t.Fatalf("should fail")
	}
	if out := ui.ErrorWriter.String(); !strings.Contains(out, dir) {
		t.Fatalf("expected mdb dir error, got: %s", out)
	}
}
