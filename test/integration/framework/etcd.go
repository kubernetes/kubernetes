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
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"go.uber.org/goleak"
	"google.golang.org/grpc/grpclog"

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

// RunCustomEtcd starts a custom etcd instance for test purposes.
func RunCustomEtcd(logger klog.Logger, dataDir string, customFlags []string) (url string, stopFn func(), err error) {
	for i := 0; i < 100; i++ {
		url, stop, err := runCustomEtcd(logger, i, dataDir, customFlags)
		if errors.Is(err, errPortInUse) {
			logger.V(0).Info("Retrying to start etcd", "attempt", i, "err", err)
			continue
		}
		return url, stop, err
	}
	return "", nil, errors.New("no unused port found despite retries")
}

// errPortInUse is returned by runCustomEtcd if picking a free port failed.
var errPortInUse = errors.New("bind: address already in use")

func runCustomEtcd(logger klog.Logger, attempt int, dataDir string, customFlags []string) (url string, stopFn func(), err error) {
	// TODO: Check for valid etcd version.
	etcdPath, err := getEtcdPath()
	if err != nil {
		fmt.Fprint(os.Stderr, installEtcd)
		return "", nil, fmt.Errorf("could not find etcd in PATH: %v", err)
	}
	etcdPort, err := getAvailablePort()
	if err != nil {
		return "", nil, fmt.Errorf("could not get a port: %v", err)
	}
	customURL := fmt.Sprintf("http://127.0.0.1:%d", etcdPort)

	// Uncomment this to test the retry mechanism reliably with
	//   go test ./test/integration/etcd/start
	//
	// if attempt == 0 {
	// 	l, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", etcdPort))
	// 	if err != nil {
	// 		return "", nil, fmt.Errorf("fake listen on etcd port: %w", err)
	// 	}
	// 	if err == nil {
	// 		defer l.Close()
	// 	}
	// }

	etcdDataDir, err := os.MkdirTemp(os.TempDir(), dataDir)
	if err != nil {
		return "", nil, fmt.Errorf("unable to make temp etcd data dir %s: %v", dataDir, err)
	}

	logger.V(2).Info("starting etcd", "url", customURL, "dataDir", etcdDataDir)
	ctx, cancel := context.WithCancel(context.Background())
	args := []string{
		"--data-dir",
		etcdDataDir,
		"--listen-client-urls",
		customURL,
		"--advertise-client-urls",
		customURL,
		"--listen-peer-urls",
		"http://127.0.0.1:0",
		"-log-level",
		"warn", // set to info or debug for more logs
		"--quota-backend-bytes",
		strconv.FormatInt(8*1024*1024*1024, 10),
	}
	args = append(args, customFlags...)
	cmd := exec.CommandContext(ctx, etcdPath, args...)

	// Always filter the etcd output. This allows us to get rid of known harmless messages and (more importantly)
	// to detect the "listen tcp 127.0.0.1:37803: bind: address already in use" error.
	reader, err := cmd.StderrPipe()
	if err != nil {
		return "", nil, fmt.Errorf("prepared etcd command's stderr pipe: %w", err)
	}
	var (
		etcdReady       = 1
		etcdBindFailure = 2
		etcdTerminated  = 3
	)
	etcdStatus := make(chan int, 2) // Large enough for initial state and final message.
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer func() {
			// If we get here, all output is processed, which means etcd must have stopped.
			etcdStatus <- etcdTerminated
		}()

		buffer := make([]byte, 100*1024)
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
						logger.Info("etcd output", "msg", string(buffer[0:n]))
					}
					continue
				}

				// Known harmless message, no need to dump it.
				msgStr, _ := msg["msg"].(string)
				switch msgStr {
				case "Running http and grpc server on single port. This is not recommended for production.":
					// Harmless...
					continue
				case "simple token is not cryptographically signed":
					// HACK! It is normal to get this warning message, but only
					// if the server actually gets passed listening on its port.
					// It would be much nicer to have some other, well-defined
					// message for that.
					//
					// There is this:
					//    {"level":"info","ts":"2025-04-11T15:36:34.680588+0200","caller":"embed/serve.go:103","msg":"ready to serve client requests"}
					//
					// But it only gets printed at info level and enabling that produced
					// also other, undesirable output like gRPC logs:
					//    2025/04/11 15:21:02 WARNING: [core] [Channel #3 SubChannel #4] grpc: addrConn.createTransport failed to connect to {Addr: "127.0.0.1:40239", ServerName: "127.0.0.1:40239", }. Err: connection error: desc = "error reading server preface: EOF"
					etcdStatus <- etcdReady
					continue
				case "ready to serve client requests":
					// We don't actually get here without -log-level info.
					etcdStatus <- etcdReady
				case "discovery failed", "failed to start etcd":
					errStr, _ := msg["error"].(string)
					if strings.Contains(errStr, "bind: address already in use") {
						// Don't dump it. Will be recovered via retries.
						etcdStatus <- etcdBindFailure
						continue
					}
				}

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

	// We need to wait until etcd is ready. Connecting to its port is not sufficient for that
	// because something else may have grabbed the port before our etcd instance did.
	// Instead we rely on etcd output to determine when it is ready.
	timeout := 30 * time.Second
	select {
	case <-time.After(timeout):
		stop()
		return "", nil, fmt.Errorf("timed waiting to etcd to start after %s", timeout)
	case status := <-etcdStatus:
		switch status {
		case etcdReady:
			return customURL, stop, nil
		case etcdTerminated:
			stop()
			return "", nil, errors.New("etcd terminated unexpectedly")
		case etcdBindFailure:
			stop()
			return "", nil, fmt.Errorf("etcd failed to start: %w", errPortInUse)
		default:
			stop()
			return "", nil, fmt.Errorf("internal error, unexpected etcd status %d", status)
		}
	}
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
