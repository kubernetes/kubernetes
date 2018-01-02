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
	"os"
	"strings"
	"testing"
	"time"

	"github.com/coreos/etcd/pkg/flags"
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

// TestCtlV3DialWithHTTPScheme ensures that client handles endpoints with HTTPS scheme.
func TestCtlV3DialWithHTTPScheme(t *testing.T) {
	testCtl(t, dialWithSchemeTest, withCfg(configClientTLS))
}

func dialWithSchemeTest(cx ctlCtx) {
	cmdArgs := append(cx.prefixArgs(cx.epc.endpoints()), "put", "foo", "bar")
	if err := spawnWithExpect(cmdArgs, "OK"); err != nil {
		cx.t.Fatal(err)
	}
}

type ctlCtx struct {
	t                 *testing.T
	cfg               etcdProcessClusterConfig
	quotaBackendBytes int64
	noStrictReconfig  bool

	epc *etcdProcessCluster

	envMap map[string]struct{}

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

func withNoStrictReconfig() ctlOption {
	return func(cx *ctlCtx) { cx.noStrictReconfig = true }
}

func withFlagByEnv() ctlOption {
	return func(cx *ctlCtx) { cx.envMap = make(map[string]struct{}) }
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
	ret.cfg.noStrictReconfig = ret.noStrictReconfig

	epc, err := newEtcdProcessCluster(&ret.cfg)
	if err != nil {
		t.Fatalf("could not start etcd process cluster (%v)", err)
	}
	ret.epc = epc

	defer func() {
		os.Unsetenv("ETCDCTL_API")
		if ret.envMap != nil {
			for k := range ret.envMap {
				os.Unsetenv(k)
			}
		}
		if errC := ret.epc.Close(); errC != nil {
			t.Fatalf("error closing etcd processes (%v)", errC)
		}
	}()

	donec := make(chan struct{})
	go func() {
		defer close(donec)
		testFunc(ret)
	}()

	timeout := 2*ret.dialTimeout + time.Second
	if ret.dialTimeout == 0 {
		timeout = 30 * time.Second
	}
	select {
	case <-time.After(timeout):
		testutil.FatalStack(t, fmt.Sprintf("test timed out after %v", timeout))
	case <-donec:
	}
}

func (cx *ctlCtx) prefixArgs(eps []string) []string {
	if len(cx.epc.proxies()) > 0 { // TODO: add proxy check as in v2
		panic("v3 proxy not implemented")
	}

	fmap := make(map[string]string)
	fmap["endpoints"] = strings.Join(eps, ",")
	fmap["dial-timeout"] = cx.dialTimeout.String()
	if cx.epc.cfg.clientTLS == clientTLS {
		if cx.epc.cfg.isClientAutoTLS {
			fmap["insecure-transport"] = "false"
			fmap["insecure-skip-tls-verify"] = "true"
		} else {
			fmap["cacert"] = caPath
			fmap["cert"] = certPath
			fmap["key"] = privateKeyPath
		}
	}
	if cx.user != "" {
		fmap["user"] = cx.user + ":" + cx.pass
	}

	useEnv := cx.envMap != nil

	cmdArgs := []string{ctlBinPath}
	for k, v := range fmap {
		if useEnv {
			ek := flags.FlagToEnv("ETCDCTL", k)
			os.Setenv(ek, v)
			cx.envMap[ek] = struct{}{}
		} else {
			cmdArgs = append(cmdArgs, fmt.Sprintf("--%s=%s", k, v))
		}
	}
	return cmdArgs
}

// PrefixArgs prefixes etcdctl command.
// Make sure to unset environment variables after tests.
func (cx *ctlCtx) PrefixArgs() []string {
	return cx.prefixArgs(cx.epc.grpcEndpoints())
}

func isGRPCTimedout(err error) bool {
	return strings.Contains(err.Error(), "grpc: timed out trying to connect")
}
