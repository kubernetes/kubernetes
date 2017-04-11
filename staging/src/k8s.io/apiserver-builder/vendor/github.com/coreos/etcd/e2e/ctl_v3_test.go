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
	"os"
	"strings"
	"testing"
	"time"

	"github.com/coreos/etcd/pkg/testutil"
	"github.com/coreos/etcd/version"
)

func TestCtlV3Version(t *testing.T) { testCtl(t, versionTest) }

func versionTest(cx ctlCtx) {
	if err := ctlV3Version(cx); err != nil {
		cx.t.Fatalf("versionTest ctlV3Version error (%v)", err)
	}
}

func ctlV3Version(cx ctlCtx) error {
	cmdArgs := append(cx.PrefixArgs(), "version")
	return spawnWithExpect(cmdArgs, version.Version)
}

type ctlCtx struct {
	t                 *testing.T
	cfg               etcdProcessClusterConfig
	quotaBackendBytes int64

	epc *etcdProcessCluster

	dialTimeout time.Duration

	quorum      bool // if true, set up 3-node cluster and linearizable read
	interactive bool

	user string
	pass string

	// for compaction
	compactPhysical bool
}

type ctlOption func(*ctlCtx)

func (cx *ctlCtx) applyOpts(opts []ctlOption) {
	for _, opt := range opts {
		opt(cx)
	}
}

func withCfg(cfg etcdProcessClusterConfig) ctlOption {
	return func(cx *ctlCtx) { cx.cfg = cfg }
}

func withDialTimeout(timeout time.Duration) ctlOption {
	return func(cx *ctlCtx) { cx.dialTimeout = timeout }
}

func withQuorum() ctlOption {
	return func(cx *ctlCtx) { cx.quorum = true }
}

func withInteractive() ctlOption {
	return func(cx *ctlCtx) { cx.interactive = true }
}

func withQuota(b int64) ctlOption {
	return func(cx *ctlCtx) { cx.quotaBackendBytes = b }
}

func withCompactPhysical() ctlOption {
	return func(cx *ctlCtx) { cx.compactPhysical = true }
}

func testCtl(t *testing.T, testFunc func(ctlCtx), opts ...ctlOption) {
	defer testutil.AfterTest(t)

	ret := ctlCtx{
		t:           t,
		cfg:         configAutoTLS,
		dialTimeout: 7 * time.Second,
	}
	ret.applyOpts(opts)

	os.Setenv("ETCDCTL_API", "3")
	mustEtcdctl(t)
	if !ret.quorum {
		ret.cfg = *configStandalone(ret.cfg)
	}
	if ret.quotaBackendBytes > 0 {
		ret.cfg.quotaBackendBytes = ret.quotaBackendBytes
	}

	epc, err := newEtcdProcessCluster(&ret.cfg)
	if err != nil {
		t.Fatalf("could not start etcd process cluster (%v)", err)
	}
	ret.epc = epc

	defer func() {
		os.Unsetenv("ETCDCTL_API")
		if errC := ret.epc.Close(); errC != nil {
			t.Fatalf("error closing etcd processes (%v)", errC)
		}
	}()

	donec := make(chan struct{})
	go func() {
		defer close(donec)
		testFunc(ret)
	}()

	select {
	case <-time.After(2*ret.dialTimeout + time.Second):
		if ret.dialTimeout > 0 {
			t.Fatalf("test timed out for %v", ret.dialTimeout)
		}
	case <-donec:
	}
}

func (cx *ctlCtx) PrefixArgs() []string {
	if len(cx.epc.proxies()) > 0 { // TODO: add proxy check as in v2
		panic("v3 proxy not implemented")
	}

	endpoints := ""
	if backends := cx.epc.backends(); len(backends) != 0 {
		es := []string{}
		for _, b := range backends {
			es = append(es, stripSchema(b.cfg.acurl))
		}
		endpoints = strings.Join(es, ",")
	}
	cmdArgs := []string{"../bin/etcdctl", "--endpoints", endpoints, "--dial-timeout", cx.dialTimeout.String()}
	if cx.epc.cfg.clientTLS == clientTLS {
		if cx.epc.cfg.isClientAutoTLS {
			cmdArgs = append(cmdArgs, "--insecure-transport=false", "--insecure-skip-tls-verify")
		} else {
			cmdArgs = append(cmdArgs, "--cacert", caPath, "--cert", certPath, "--key", privateKeyPath)
		}
	}

	if cx.user != "" {
		cmdArgs = append(cmdArgs, "--user="+cx.user+":"+cx.pass)
	}

	return cmdArgs
}

func isGRPCTimedout(err error) bool {
	return strings.Contains(err.Error(), "grpc: timed out trying to connect")
}

func stripSchema(s string) string {
	if strings.HasPrefix(s, "http://") {
		s = strings.Replace(s, "http://", "", -1)
	}
	if strings.HasPrefix(s, "https://") {
		s = strings.Replace(s, "https://", "", -1)
	}
	return s
}
