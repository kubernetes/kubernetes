// Copyright 2016 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package e2e

import (
	"fmt"
	"io/ioutil"
	"net/url"
	"os"
	"strings"

	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/pkg/expect"
	"github.com/coreos/etcd/pkg/fileutil"
)

const etcdProcessBasePort = 20000

var (
	binPath        string
	ctlBinPath     string
	certPath       string
	privateKeyPath string
	caPath         string
)

type clientConnType int

const (
	clientNonTLS clientConnType = iota
	clientTLS
	clientTLSAndNonTLS
)

var (
	configNoTLS = etcdProcessClusterConfig{
		clusterSize:  3,
		proxySize:    0,
		initialToken: "new",
	}
	configAutoTLS = etcdProcessClusterConfig{
		clusterSize:   3,
		isPeerTLS:     true,
		isPeerAutoTLS: true,
		initialToken:  "new",
	}
	configTLS = etcdProcessClusterConfig{
		clusterSize:  3,
		proxySize:    0,
		clientTLS:    clientTLS,
		isPeerTLS:    true,
		initialToken: "new",
	}
	configClientTLS = etcdProcessClusterConfig{
		clusterSize:  3,
		proxySize:    0,
		clientTLS:    clientTLS,
		initialToken: "new",
	}
	configClientBoth = etcdProcessClusterConfig{
		clusterSize:  1,
		proxySize:    0,
		clientTLS:    clientTLSAndNonTLS,
		initialToken: "new",
	}
	configClientAutoTLS = etcdProcessClusterConfig{
		clusterSize:     1,
		proxySize:       0,
		isClientAutoTLS: true,
		clientTLS:       clientTLS,
		initialToken:    "new",
	}
	configPeerTLS = etcdProcessClusterConfig{
		clusterSize:  3,
		proxySize:    0,
		isPeerTLS:    true,
		initialToken: "new",
	}
	configWithProxy = etcdProcessClusterConfig{
		clusterSize:  3,
		proxySize:    1,
		initialToken: "new",
	}
	configWithProxyTLS = etcdProcessClusterConfig{
		clusterSize:  3,
		proxySize:    1,
		clientTLS:    clientTLS,
		isPeerTLS:    true,
		initialToken: "new",
	}
	configWithProxyPeerTLS = etcdProcessClusterConfig{
		clusterSize:  3,
		proxySize:    1,
		isPeerTLS:    true,
		initialToken: "new",
	}
)

func configStandalone(cfg etcdProcessClusterConfig) *etcdProcessClusterConfig {
	ret := cfg
	ret.clusterSize = 1
	return &ret
}

type etcdProcessCluster struct {
	cfg   *etcdProcessClusterConfig
	procs []*etcdProcess
}

type etcdProcess struct {
	cfg   *etcdProcessConfig
	proc  *expect.ExpectProcess
	donec chan struct{} // closed when Interact() terminates
}

type etcdProcessConfig struct {
	execPath string
	args     []string

	dataDirPath string
	keepDataDir bool

	name string

	purl url.URL

	acurl string
	// additional url for tls connection when the etcd process
	// serves both http and https
	acurltls  string
	acurlHost string

	initialToken   string
	initialCluster string

	isProxy bool
}

type etcdProcessClusterConfig struct {
	execPath    string
	dataDirPath string
	keepDataDir bool

	clusterSize int

	baseScheme string
	basePort   int

	proxySize int

	snapCount int // default is 10000

	clientTLS             clientConnType
	clientCertAuthEnabled bool
	isPeerTLS             bool
	isPeerAutoTLS         bool
	isClientAutoTLS       bool
	forceNewCluster       bool
	initialToken          string
	quotaBackendBytes     int64
	noStrictReconfig      bool
}

// newEtcdProcessCluster launches a new cluster from etcd processes, returning
// a new etcdProcessCluster once all nodes are ready to accept client requests.
func newEtcdProcessCluster(cfg *etcdProcessClusterConfig) (*etcdProcessCluster, error) {
	etcdCfgs := cfg.etcdProcessConfigs()
	epc := &etcdProcessCluster{
		cfg:   cfg,
		procs: make([]*etcdProcess, cfg.clusterSize+cfg.proxySize),
	}

	// launch etcd processes
	for i := range etcdCfgs {
		proc, err := newEtcdProcess(etcdCfgs[i])
		if err != nil {
			epc.Close()
			return nil, err
		}
		epc.procs[i] = proc
	}

	return epc, epc.Start()
}

func newEtcdProcess(cfg *etcdProcessConfig) (*etcdProcess, error) {
	if !fileutil.Exist(cfg.execPath) {
		return nil, fmt.Errorf("could not find etcd binary")
	}

	if !cfg.keepDataDir {
		if err := os.RemoveAll(cfg.dataDirPath); err != nil {
			return nil, err
		}
	}

	child, err := spawnCmd(append([]string{cfg.execPath}, cfg.args...))
	if err != nil {
		return nil, err
	}
	return &etcdProcess{cfg: cfg, proc: child, donec: make(chan struct{})}, nil
}

func (cfg *etcdProcessClusterConfig) etcdProcessConfigs() []*etcdProcessConfig {
	binPath = binDir + "/etcd"
	ctlBinPath = binDir + "/etcdctl"
	certPath = certDir + "/server.crt"
	privateKeyPath = certDir + "/server.key.insecure"
	caPath = certDir + "/ca.crt"

	if cfg.basePort == 0 {
		cfg.basePort = etcdProcessBasePort
	}

	if cfg.execPath == "" {
		cfg.execPath = binPath
	}
	if cfg.snapCount == 0 {
		cfg.snapCount = etcdserver.DefaultSnapCount
	}

	clientScheme := "http"
	if cfg.clientTLS == clientTLS {
		clientScheme = "https"
	}
	peerScheme := cfg.baseScheme
	if peerScheme == "" {
		peerScheme = "http"
	}
	if cfg.isPeerTLS {
		peerScheme += "s"
	}

	etcdCfgs := make([]*etcdProcessConfig, cfg.clusterSize+cfg.proxySize)
	initialCluster := make([]string, cfg.clusterSize)
	for i := 0; i < cfg.clusterSize; i++ {
		var curls []string
		var curl, curltls string
		port := cfg.basePort + 2*i
		curlHost := fmt.Sprintf("localhost:%d", port)

		switch cfg.clientTLS {
		case clientNonTLS, clientTLS:
			curl = (&url.URL{Scheme: clientScheme, Host: curlHost}).String()
			curls = []string{curl}
		case clientTLSAndNonTLS:
			curl = (&url.URL{Scheme: "http", Host: curlHost}).String()
			curltls = (&url.URL{Scheme: "https", Host: curlHost}).String()
			curls = []string{curl, curltls}
		}

		purl := url.URL{Scheme: peerScheme, Host: fmt.Sprintf("localhost:%d", port+1)}
		name := fmt.Sprintf("testname%d", i)
		dataDirPath := cfg.dataDirPath
		if cfg.dataDirPath == "" {
			var derr error
			dataDirPath, derr = ioutil.TempDir("", name+".etcd")
			if derr != nil {
				panic("could not get tempdir for datadir")
			}
		}
		initialCluster[i] = fmt.Sprintf("%s=%s", name, purl.String())

		args := []string{
			"--name", name,
			"--listen-client-urls", strings.Join(curls, ","),
			"--advertise-client-urls", strings.Join(curls, ","),
			"--listen-peer-urls", purl.String(),
			"--initial-advertise-peer-urls", purl.String(),
			"--initial-cluster-token", cfg.initialToken,
			"--data-dir", dataDirPath,
			"--snapshot-count", fmt.Sprintf("%d", cfg.snapCount),
		}
		if cfg.forceNewCluster {
			args = append(args, "--force-new-cluster")
		}
		if cfg.quotaBackendBytes > 0 {
			args = append(args,
				"--quota-backend-bytes", fmt.Sprintf("%d", cfg.quotaBackendBytes),
			)
		}
		if cfg.noStrictReconfig {
			args = append(args, "--strict-reconfig-check=false")
		}

		args = append(args, cfg.tlsArgs()...)
		etcdCfgs[i] = &etcdProcessConfig{
			execPath:     cfg.execPath,
			args:         args,
			dataDirPath:  dataDirPath,
			keepDataDir:  cfg.keepDataDir,
			name:         name,
			purl:         purl,
			acurl:        curl,
			acurltls:     curltls,
			acurlHost:    curlHost,
			initialToken: cfg.initialToken,
		}
	}
	for i := 0; i < cfg.proxySize; i++ {
		port := cfg.basePort + 2*cfg.clusterSize + i + 1
		curlHost := fmt.Sprintf("localhost:%d", port)
		curl := url.URL{Scheme: clientScheme, Host: curlHost}
		name := fmt.Sprintf("testname-proxy%d", i)
		dataDirPath, derr := ioutil.TempDir("", name+".etcd")
		if derr != nil {
			panic("could not get tempdir for datadir")
		}
		args := []string{
			"--name", name,
			"--proxy", "on",
			"--listen-client-urls", curl.String(),
			"--data-dir", dataDirPath,
		}
		args = append(args, cfg.tlsArgs()...)
		etcdCfgs[cfg.clusterSize+i] = &etcdProcessConfig{
			execPath:    cfg.execPath,
			args:        args,
			dataDirPath: dataDirPath,
			keepDataDir: cfg.keepDataDir,
			name:        name,
			acurl:       curl.String(),
			acurlHost:   curlHost,
			isProxy:     true,
		}
	}

	initialClusterArgs := []string{"--initial-cluster", strings.Join(initialCluster, ",")}
	for i := range etcdCfgs {
		etcdCfgs[i].initialCluster = strings.Join(initialCluster, ",")
		etcdCfgs[i].args = append(etcdCfgs[i].args, initialClusterArgs...)
	}

	return etcdCfgs
}

func (cfg *etcdProcessClusterConfig) tlsArgs() (args []string) {
	if cfg.clientTLS != clientNonTLS {
		if cfg.isClientAutoTLS {
			args = append(args, "--auto-tls=true")
		} else {
			tlsClientArgs := []string{
				"--cert-file", certPath,
				"--key-file", privateKeyPath,
				"--ca-file", caPath,
			}
			args = append(args, tlsClientArgs...)

			if cfg.clientCertAuthEnabled {
				args = append(args, "--client-cert-auth")
			}
		}
	}

	if cfg.isPeerTLS {
		if cfg.isPeerAutoTLS {
			args = append(args, "--peer-auto-tls=true")
		} else {
			tlsPeerArgs := []string{
				"--peer-cert-file", certPath,
				"--peer-key-file", privateKeyPath,
				"--peer-ca-file", caPath,
			}
			args = append(args, tlsPeerArgs...)
		}
	}
	return args
}

func (epc *etcdProcessCluster) Start() (err error) {
	readyC := make(chan error, epc.cfg.clusterSize+epc.cfg.proxySize)
	for i := range epc.procs {
		go func(n int) { readyC <- epc.procs[n].waitReady() }(i)
	}
	for range epc.procs {
		if err := <-readyC; err != nil {
			epc.Close()
			return err
		}
	}
	return nil
}

func (epc *etcdProcessCluster) RestartAll() error {
	for i := range epc.procs {
		proc, err := newEtcdProcess(epc.procs[i].cfg)
		if err != nil {
			epc.Close()
			return err
		}
		epc.procs[i] = proc
	}
	return epc.Start()
}

func (epc *etcdProcessCluster) StopAll() (err error) {
	for _, p := range epc.procs {
		if p == nil {
			continue
		}
		if curErr := p.Stop(); curErr != nil {
			if err != nil {
				err = fmt.Errorf("%v; %v", err, curErr)
			} else {
				err = curErr
			}
		}
	}
	return err
}

func (epc *etcdProcessCluster) Close() error {
	err := epc.StopAll()
	for _, p := range epc.procs {
		// p is nil when newEtcdProcess fails in the middle
		// Close still gets called to clean up test data
		if p == nil {
			continue
		}
		os.RemoveAll(p.cfg.dataDirPath)
	}
	return err
}

func (ep *etcdProcess) Restart() error {
	newEp, err := newEtcdProcess(ep.cfg)
	if err != nil {
		ep.Stop()
		return err
	}
	*ep = *newEp
	if err = ep.waitReady(); err != nil {
		ep.Stop()
		return err
	}
	return nil
}

func (ep *etcdProcess) Stop() error {
	if ep == nil {
		return nil
	}
	if err := ep.proc.Stop(); err != nil {
		return err
	}
	<-ep.donec

	if ep.cfg.purl.Scheme == "unix" || ep.cfg.purl.Scheme == "unixs" {
		os.RemoveAll(ep.cfg.purl.Host)
	}
	return nil
}

func (ep *etcdProcess) waitReady() error {
	defer close(ep.donec)
	return waitReadyExpectProc(ep.proc, ep.cfg.isProxy)
}

func waitReadyExpectProc(exproc *expect.ExpectProcess, isProxy bool) error {
	readyStrs := []string{"enabled capabilities for version", "published"}
	if isProxy {
		readyStrs = []string{"httpproxy: endpoints found"}
	}
	c := 0
	matchSet := func(l string) bool {
		for _, s := range readyStrs {
			if strings.Contains(l, s) {
				c++
				break
			}
		}
		return c == len(readyStrs)
	}
	_, err := exproc.ExpectFunc(matchSet)
	return err
}

func spawnCmd(args []string) (*expect.ExpectProcess, error) {
	return expect.NewExpect(args[0], args[1:]...)
}

func spawnWithExpect(args []string, expected string) error {
	return spawnWithExpects(args, []string{expected}...)
}

func spawnWithExpects(args []string, xs ...string) error {
	proc, err := spawnCmd(args)
	if err != nil {
		return err
	}
	// process until either stdout or stderr contains
	// the expected string
	var (
		lines    []string
		lineFunc = func(txt string) bool { return true }
	)
	for _, txt := range xs {
		for {
			l, err := proc.ExpectFunc(lineFunc)
			if err != nil {
				proc.Close()
				return fmt.Errorf("%v (expected %q, got %q)", err, txt, lines)
			}
			lines = append(lines, l)
			if strings.Contains(l, txt) {
				break
			}
		}
	}
	perr := proc.Close()
	if err != nil {
		return err
	}
	if len(xs) == 0 && proc.LineCount() != 0 { // expect no output
		return fmt.Errorf("unexpected output (got lines %q, line count %d)", lines, proc.LineCount())
	}
	return perr
}

// proxies returns only the proxy etcdProcess.
func (epc *etcdProcessCluster) proxies() []*etcdProcess {
	return epc.procs[epc.cfg.clusterSize:]
}

func (epc *etcdProcessCluster) processes() []*etcdProcess {
	return epc.procs[:epc.cfg.clusterSize]
}

func (epc *etcdProcessCluster) endpoints() []string {
	eps := make([]string, epc.cfg.clusterSize)
	for i, ep := range epc.processes() {
		eps[i] = ep.cfg.acurl
	}
	return eps
}

func (epc *etcdProcessCluster) grpcEndpoints() []string {
	eps := make([]string, epc.cfg.clusterSize)
	for i, ep := range epc.processes() {
		eps[i] = ep.cfg.acurlHost
	}
	return eps
}
