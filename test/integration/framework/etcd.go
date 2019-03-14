/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package framework

import (
	"context"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"k8s.io/klog"

	"k8s.io/kubernetes/pkg/util/env"
)

var etcdURL = ""

const installEtcd = `
Cannot find etcd, cannot run integration tests
Please see https://git.k8s.io/community/contributors/devel/sig-testing/testing.md#install-etcd-dependency for instructions.

You can use 'hack/install-etcd.sh' to install a copy in third_party/.

`

// getEtcdPath returns a path to an etcd executable.
func getEtcdPath() (string, error) {
	bazelPath := filepath.Join(os.Getenv("RUNFILES_DIR"), fmt.Sprintf("com_coreos_etcd_%s", runtime.GOARCH), "etcd")

	p, err := exec.LookPath(bazelPath)
	if err == nil {
		return p, nil
	}
	return exec.LookPath("etcd")
}

// getAvailablePort returns a TCP port that is available for binding.
func getAvailablePort() (int, error) {
	l, err := net.Listen("tcp", ":0")
	if err != nil {
		return 0, fmt.Errorf("could not bind to a port: %v", err)
	}
	// It is possible but unlikely that someone else will bind this port before we
	// get a chance to use it.
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

// startEtcd executes an etcd instance. The returned function will signal the
// etcd process and wait for it to exit.
func startEtcd() (func(), error) {
	etcdURL = env.GetEnvAsStringOrFallback("KUBE_INTEGRATION_ETCD_URL", "http://127.0.0.1:2379")
	conn, err := net.Dial("tcp", strings.TrimPrefix(etcdURL, "http://"))
	if err == nil {
		klog.Infof("etcd already running at %s", etcdURL)
		conn.Close()
		return func() {}, nil
	}
	klog.V(1).Infof("could not connect to etcd: %v", err)

	// TODO: Check for valid etcd version.
	etcdPath, err := getEtcdPath()
	if err != nil {
		fmt.Fprintf(os.Stderr, installEtcd)
		return nil, fmt.Errorf("could not find etcd in PATH: %v", err)
	}
	etcdPort, err := getAvailablePort()
	if err != nil {
		return nil, fmt.Errorf("could not get a port: %v", err)
	}
	etcdURL = fmt.Sprintf("http://127.0.0.1:%d", etcdPort)
	klog.Infof("starting etcd on %s", etcdURL)

	etcdDataDir, err := ioutil.TempDir(os.TempDir(), "integration_test_etcd_data")
	if err != nil {
		return nil, fmt.Errorf("unable to make temp etcd data dir: %v", err)
	}
	klog.Infof("storing etcd data in: %v", etcdDataDir)

	ctx, cancel := context.WithCancel(context.Background())
	cmd := exec.CommandContext(
		ctx,
		etcdPath,
		"--data-dir",
		etcdDataDir,
		"--listen-client-urls",
		GetEtcdURL(),
		"--advertise-client-urls",
		GetEtcdURL(),
		"--listen-peer-urls",
		"http://127.0.0.1:0",
		"--log-package-levels",
		"*=DEBUG",
	)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	stop := func() {
		cancel()
		err := cmd.Wait()
		klog.Infof("etcd exit status: %v", err)
		err = os.RemoveAll(etcdDataDir)
		if err != nil {
			klog.Warningf("error during etcd cleanup: %v", err)
		}
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to run etcd: %v", err)
	}
	return stop, nil
}

// EtcdMain starts an etcd instance before running tests.
func EtcdMain(tests func() int) {
	stop, err := startEtcd()
	if err != nil {
		klog.Fatalf("cannot run integration tests: unable to start etcd: %v", err)
	}
	result := tests()
	stop() // Don't defer this. See os.Exit documentation.
	os.Exit(result)
}

// GetEtcdURL returns the URL of the etcd instance started by EtcdMain.
func GetEtcdURL() string {
	return etcdURL
}
