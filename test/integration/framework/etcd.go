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
	"fmt"
	"hash/adler32"
	"io"
	"io/ioutil"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"sync"

	"k8s.io/kubernetes/pkg/util/env"

	"github.com/golang/glog"
)

var (
	etcdSetup sync.Once
	etcdURL   = ""
)

func setupETCD() {
	etcdSetup.Do(func() {
		if os.Getenv("RUNFILES_DIR") == "" {
			etcdURL = env.GetEnvAsStringOrFallback("KUBE_INTEGRATION_ETCD_URL", "http://127.0.0.1:2379")
			return
		}
		etcdPath := filepath.Join(os.Getenv("RUNFILES_DIR"), "com_coreos_etcd/etcd")
		// give every test the same random port each run
		etcdPort := 20000 + rand.New(rand.NewSource(int64(adler32.Checksum([]byte(os.Args[0]))))).Intn(5000)
		etcdURL = fmt.Sprintf("http://127.0.0.1:%d", etcdPort)

		info, err := os.Stat(etcdPath)
		if err != nil {
			glog.Fatalf("Unable to stat etcd: %v", err)
		}
		if info.IsDir() {
			glog.Fatalf("Did not expect %q to be a directory", etcdPath)
		}

		etcdDataDir, err := ioutil.TempDir(os.TempDir(), "integration_test_etcd_data")
		if err != nil {
			glog.Fatalf("Unable to make temp etcd data dir: %v", err)
		}
		glog.Infof("storing etcd data in: %v", etcdDataDir)

		etcdCmd := exec.Command(
			etcdPath,
			"--data-dir",
			etcdDataDir,
			"--listen-client-urls",
			GetEtcdURL(),
			"--advertise-client-urls",
			GetEtcdURL(),
			"--listen-peer-urls",
			"http://127.0.0.1:0",
		)

		stdout, err := etcdCmd.StdoutPipe()
		if err != nil {
			glog.Fatalf("Failed to run etcd: %v", err)
		}
		stderr, err := etcdCmd.StderrPipe()
		if err != nil {
			glog.Fatalf("Failed to run etcd: %v", err)
		}
		if err := etcdCmd.Start(); err != nil {
			glog.Fatalf("Failed to run etcd: %v", err)
		}

		go io.Copy(os.Stdout, stdout)
		go io.Copy(os.Stderr, stderr)

		go func() {
			if err := etcdCmd.Wait(); err != nil {
				glog.Fatalf("Failed to run etcd: %v", err)
			}
			glog.Fatalf("etcd should not have succeeded")
		}()
	})
}

func EtcdMain(tests func() int) {
	setupETCD()
	os.Exit(tests())
}

// return the EtcdURL
func GetEtcdURL() string {
	return etcdURL
}
