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

const (
	etcdProcessBasePort = 20000
	certPath            = "../integration/fixtures/server.crt"
	privateKeyPath      = "../integration/fixtures/server.key.insecure"
	caPath              = "../integration/fixtures/ca.crt"
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

	acurl string
	// additional url for tls connection when the etcd process
	// serves both http and https
	acurltls  string
	acurlHost string

	isProxy bool
}

type etcdProcessClusterConfig struct {
	execPath    string
	dataDirPath string
	keepDataDir bool

	clusterSize int
	basePort    int
	proxySize   int

	snapCount int // default is 10000

	clientTLS             clientConnType
	clientCertAuthEnabled bool
	isPeerTLS             bool
	isPeerAutoTLS         bool
	isClientAutoTLS       bool
	forceNewCluster       bool
	initialToken          string
	quotaBackendBytes     int64
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
	if cfg.basePort == 0 {
		cfg.basePort = etcdProcessBasePort
	}

	if cfg.execPath == "" {
		cfg.execPath = "../bin/etcd"
	}
	if cfg.snapCount == 0 {
		cfg.snapCount = etcdserver.DefaultSnapCount
	}

	clientScheme := "http"
	if cfg.clientTLS == clientTLS {
		clientScheme = "https"
	}
	peerScheme := "http"
	if cfg.isPeerTLS {
		peerScheme = "https"
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

		args = append(args, cfg.tlsArgs()...)
		etcdCfgs[i] = &etcdProcessConfig{
			execPath:    cfg.execPath,
			args:        args,
			dataDirPath: dataDirPath,
			keepDataDir: cfg.keepDataDir,
			acurl:       curl,
			acurltls:    curltls,
			acurlHost:   curlHost,
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
			acurl:       curl.String(),
			acurlHost:   curlHost,
			isProxy:     true,
		}
	}

	initialClusterArgs := []string{"--initial-cluster", strings.Join(initialCluster, ",")}
	for i := range etcdCfgs {
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
	readyStr := "enabled capabilities for version"
	for i := range epc.procs {
		go func(etcdp *etcdProcess) {
			etcdp.donec = make(chan struct{})
			rs := readyStr
			if etcdp.cfg.isProxy {
				rs = "httpproxy: endpoints found"
			}
			_, err := etcdp.proc.Expect(rs)
			readyC <- err
			close(etcdp.donec)
		}(epc.procs[i])
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

func (epr *etcdProcess) Restart() error {
	proc, err := newEtcdProcess(epr.cfg)
	if err != nil {
		epr.Stop()
		return err
	}
	*epr = *proc

	readyStr := "enabled capabilities for version"
	if proc.cfg.isProxy {
		readyStr = "httpproxy: endpoints found"
	}

	if _, err = proc.proc.Expect(readyStr); err != nil {
		epr.Stop()
		return err
	}
	close(proc.donec)

	return nil
}

func (epc *etcdProcessCluster) StopAll() (err error) {
	for _, p := range epc.procs {
		if p == nil {
			continue
		}
		if curErr := p.proc.Stop(); curErr != nil {
			if err != nil {
				err = fmt.Errorf("%v; %v", err, curErr)
			} else {
				err = curErr
			}
		}
		<-p.donec
	}
	return err
}

func (epr *etcdProcess) Stop() error {
	if epr == nil {
		return nil
	}

	if err := epr.proc.Stop(); err != nil {
		return err
	}

	<-epr.donec
	return nil
}

func (epc *etcdProcessCluster) Close() error {
	err := epc.StopAll()
	for _, p := range epc.procs {
		os.RemoveAll(p.cfg.dataDirPath)
	}
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

func (epc *etcdProcessCluster) backends() []*etcdProcess {
	return epc.procs[:epc.cfg.clusterSize]
}

func (epc *etcdProcessCluster) endpoints() []string {
	eps := make([]string, epc.cfg.clusterSize)
	for i, ep := range epc.backends() {
		eps[i] = ep.cfg.acurl
	}
	return eps
}

func (epc *etcdProcessCluster) grpcEndpoints() []string {
	eps := make([]string, epc.cfg.clusterSize)
	for i, ep := range epc.backends() {
		eps[i] = ep.cfg.acurlHost
	}
	return eps
}
