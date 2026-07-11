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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"go.uber.org/goleak"
	"google.golang.org/grpc/grpclog"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/util/env"
)

const installEtcd = `
Cannot find etcd, cannot run integration tests
Please see https://git.k8s.io/community/contributors/devel/sig-testing/integration-tests.md#install-etcd-dependency for instructions.

You can use 'hack/install-etcd.sh' to install a copy in third_party/.

`

// getEtcdPath returns a path to an etcd executable.
func getEtcdPath() (string, error) {
	return exec.LookPath("etcd")
}

// startEtcd executes an etcd instance. The returned function will signal the
// etcd process and wait for it to exit.
func startEtcd(logger klog.Logger, forceCreate bool) (func(), error) {
	if !forceCreate {
		etcdURL := env.GetEnvAsStringOrFallback("KUBE_INTEGRATION_ETCD_URL", "http://127.0.0.1:2379")
		conn, err := net.Dial("tcp", strings.TrimPrefix(etcdURL, "http://"))
		if err == nil {
			klog.Infof("etcd already running at %s", etcdURL)
			_ = conn.Close()
			return func() {}, nil
		}
		logger.V(1).Info("could not connect to etcd", "err", err)
	}

	currentURL, stop, err := RunCustomEtcd(logger, "integration_test_etcd_data", nil)
	if err != nil {
		return nil, err
	}

	os.Setenv("KUBE_INTEGRATION_ETCD_URL", currentURL)

	return stop, nil
}

func init() {
	// Quiet etcd logs for integration tests
	// Comment out to get verbose logs if desired.
	// This has to be done before there are any goroutines
	// active which use gRPC. During init is safe, albeit
	// then also affects tests which don't use RunCustomEtcd
	// (the place this was done before).
	grpclog.SetLoggerV2(grpclog.NewLoggerV2(io.Discard, io.Discard, os.Stderr))
}

// hiddenEtcdMessages contains all messages that occur due to the way we run
// etcd and don't need to be logged in all tests.
var hiddenEtcdMessages = sets.New(
	"simple token is not cryptographically signed",
	"Running http and grpc server on single port. This is not recommended for production.",
)

// RunCustomEtcd starts a custom etcd instance for test purposes.
func RunCustomEtcd(logger klog.Logger, dataDir string, customFlags []string) (url string, stopFn func(), err error) {
	// TODO: Check for valid etcd version.
	etcdPath, err := getEtcdPath()
	if err != nil {
		fmt.Fprint(os.Stderr, installEtcd)
		return "", nil, fmt.Errorf("could not find etcd in PATH: %v", err)
	}
	etcdDataDir, err := os.MkdirTemp(os.TempDir(), dataDir)
	if err != nil {
		return "", nil, fmt.Errorf("unable to make temp etcd data dir %s: %v", dataDir, err)
	}
	etcdSocketPath := path.Join(etcdDataDir, "etcd.sock")
	customURL := "unix://" + etcdSocketPath

	logger.V(2).Info("starting etcd", "url", customURL, "dataDir", etcdDataDir)
	ctx, cancel := context.WithCancel(context.Background())
	args := []string{
		"--data-dir",
		etcdDataDir,
		"--listen-client-urls",
		customURL,
		// This should be how clients connect to etcd, but https://github.com/etcd-io/etcd/pull/12469
		// apparently was incomplete: trying to pass a Unix Domain URL here is rejected by ectd 3.15.13 with
		//    --advertise-client-urls "unix:///tmp/etcd.sock" must be "host:port" (missing port in address)
		//
		// We don't need to advertise the correct address. To prevent connecting to the default URL
		// in the unlikely case that something does use this URL after all, an invalid URL is set here.
		"--advertise-client-urls",
		"http://127.0.0.111:0",
		// With :0 we let the kernel pick a unique port. We don't care which port this will be,
		// no other peer is going to connect.
		"--listen-peer-urls",
		"http://127.0.0.1:0",
		"-log-level",
		"warn", // set to info or debug for more logs
		"--quota-backend-bytes",
		strconv.FormatInt(8*1024*1024*1024, 10),
	}
	args = append(args, customFlags...)
	cmd := exec.CommandContext(ctx, etcdPath, args...)

	// Always filter the etcd output. This allows us to get rid of known harmless messages
	// and includes the etcd log messages in the normal test log.
	reader, err := cmd.StderrPipe()
	if err != nil {
		cancel()
		return "", nil, fmt.Errorf("prepared etcd command's stderr pipe: %w", err)
	}
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		buffer := make([]byte, 100*1024)
	reading:
		for {
			n, err := reader.Read(buffer)
			// Unfortunately in practice we get an untyped errors.errorString wrapped in an os.Path error,
			// so we have to fall back to text matching.
			if errors.Is(err, io.EOF) || err != nil && err.Error() == "read |0: file already closed" {
				return
			}
			if err != nil {
				logger.Error(err, "read etcd output")
				return
			}
			if n == 0 {
				continue
			}
			dec := json.NewDecoder(bytes.NewBuffer(buffer[0:n]))
			for {
				// Try to parse as JSON object. If we get anything that isn't JSON or an object, we just dump the remaining output.
				var msg map[string]any
				err := dec.Decode(&msg)
				if err == io.EOF {
					// all done
					break
				}
				if err != nil {
					offset := int(dec.InputOffset())
					if offset < n {
						logger.Info("etcd output", "msg", string(buffer[offset:n]))
					}
					continue reading
				}

				// Skip harmless messages.
				msgStr, _ := msg["msg"].(string)
				if hiddenEtcdMessages.Has(msgStr) {
					continue
				}

				// Log it using structured logging.
				kvs := make([]any, 0, 2*len(msg))
				for key, value := range msg {
					kvs = append(kvs, key, value)
				}
				logger.Info("etcd output", kvs...)
			}
		}
	}()

	stop := func() {
		// try to exit etcd gracefully
		defer cancel()
		cmd.Process.Signal(syscall.SIGTERM)
		wg.Add(1)
		go func() {
			defer wg.Done()
			select {
			case <-ctx.Done():
				logger.V(6).Info("etcd exited gracefully, context cancelled")
			case <-time.After(5 * time.Second):
				logger.Info("etcd didn't exit in 5 seconds, killing it")
				cancel()
			}
		}()
		err := cmd.Wait()
		logger.V(2).Info("etcd exited", "err", err)
		// Tell goroutine that we are done.
		cancel()
		wg.Wait()
		err = os.RemoveAll(etcdDataDir)
		if err != nil {
			logger.Info("Warning: error during etcd cleanup", "err", err)
		}
	}

	if err := cmd.Start(); err != nil {
		return "", nil, fmt.Errorf("failed to run etcd: %v", err)
	}

	var i int32 = 1
	const pollCount = int32(300)

	for i <= pollCount {
		conn, err := net.DialTimeout("unix", etcdSocketPath, 1*time.Second)
		if err == nil {
			conn.Close()
			break
		}

		if i == pollCount {
			stop()
			return "", nil, fmt.Errorf("could not start etcd")
		}

		time.Sleep(100 * time.Millisecond)
		i = i + 1
	}

	return customURL, stop, nil
}

// EtcdMain starts an etcd instance before running tests.
func EtcdMain(tests func() int) {
	// Bail out early when -help was given as parameter.
	flag.Parse()

	// Must be called *before* creating new goroutines.
	goleakOpts := IgnoreBackgroundGoroutines()

	goleakOpts = append(goleakOpts,
		// lumberjack leaks a goroutine:
		// https://github.com/natefinch/lumberjack/issues/56 This affects tests
		// using --audit-log-path (like
		// ./test/integration/apiserver/admissionwebhook/reinvocation_test.go).
		// In normal production that should be harmless. We don't know here
		// whether the test is using that, so we have to suppress reporting
		// this leak for all tests.
		//
		// Both names occurred in practice.
		goleak.IgnoreTopFunction("k8s.io/kubernetes/vendor/gopkg.in/natefinch/lumberjack%2ev2.(*Logger).millRun"),
		goleak.IgnoreTopFunction("gopkg.in/natefinch/lumberjack%2ev2.(*Logger).millRun"),
		// If there is an error during connection shutdown, SPDY keeps a
		// goroutine around for ten minutes, which gets reported as a leak
		// (the goroutine is cleaned up after the ten minutes).
		//
		// https://github.com/kubernetes/kubernetes/blob/master/vendor/github.com/moby/spdystream/connection.go#L732-L744
		//
		// Ignore this reported leak.
		goleak.IgnoreTopFunction("github.com/moby/spdystream.(*Connection).shutdown"),
	)

	stop, err := startEtcd(klog.Background(), false)
	if err != nil {
		klog.Fatalf("cannot run integration tests: unable to start etcd: %v", err)
	}
	result := tests()
	stop() // Don't defer this. See os.Exit documentation.
	klog.StopFlushDaemon()

	if err := goleakFindRetry(goleakOpts...); err != nil {
		klog.ErrorS(err, "EtcdMain goroutine check")
		result = 1
	}

	os.Exit(result)
}

// GetEtcdURL returns the URL of the etcd instance started by EtcdMain or StartEtcd.
func GetEtcdURL() string {
	return env.GetEnvAsStringOrFallback("KUBE_INTEGRATION_ETCD_URL", "http://127.0.0.1:2379")
}

// StartEtcd starts an etcd instance inside a test. It will abort the test if
// startup fails and clean up after the test automatically. Stdout and stderr
// of the etcd binary and all log output of the etcd helper code go to the
// provided logger.
//
// In contrast to EtcdMain, StartEtcd will not do automatic leak checking.
// Tests can decide if and where they want to do that.
//
// Starting etcd multiple times per test run instead of once with EtcdMain
// provides better separation between different tests.
func StartEtcd(logger klog.Logger, tb testing.TB, forceCreate bool) {
	stop, err := startEtcd(logger, forceCreate)
	if err != nil {
		tb.Fatalf("unable to start etcd: %v", err)
	}
	tb.Cleanup(stop)
}
