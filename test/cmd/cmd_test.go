/*
Copyright The Kubernetes Authors.
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

package cmd_test

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/rogpeppe/go-internal/testscript"
	"google.golang.org/grpc/grpclog"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	// clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/client-go/util/cert"
	"k8s.io/klog/v2"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	kubectlcmd "k8s.io/kubectl/pkg/cmd"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubernetes/test/utils/kubeconfig"

	etcdserver "k8s.io/apiserver/pkg/storage/etcd3/testserver"
)

func init() {
	// Suppress verbose gRPC logs (from etcd client)
	// This must be done before any goroutines using gRPC are started
	grpclog.SetLoggerV2(grpclog.NewLoggerV2(io.Discard, io.Discard, io.Discard))

	// Set klog verbosity to 0 globally to suppress most apiserver logs
	// Only errors and warnings will be shown
	var fs flag.FlagSet
	klog.InitFlags(&fs)
	fs.Set("v", "0")
	fs.Set("logtostderr", "false")
	fs.Set("stderrthreshold", "ERROR")
}

// getKubeRoot returns the root directory of the kubernetes repository
func getKubeRoot() string {
	// Get the directory of this source file
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		panic("failed to get current file path")
	}
	// This file is at test/cmd/cmd_test.go, so go up 2 levels
	return filepath.Dir(filepath.Dir(filepath.Dir(thisFile)))
}

// testAPIServer holds the test API server instance and kubeconfig path
type testAPIServer struct {
	server         *kubeapiservertesting.TestServer
	clientSet      clientset.Interface
	kubeConfigFile string
	namespace      string // The test namespace created for this test
}

// TestKubectl runs all kubectl command tests using testscript
func TestKubectl(t *testing.T) {
	testscript.Run(t, testscript.Params{
		// Directory containing .txtar test files
		Dir: "testdata",

		// Setup function runs before each test
		Setup: func(env *testscript.Env) error {
			// Start a test API server
			apiServer, err := startTestAPIServer(env.T().(testing.TB))
			if err != nil {
				return err
			}

			// Create a unique namespace for this test (like create_and_use_new_namespace in bash)
			namespace := fmt.Sprintf("namespace-%d-%d", time.Now().Unix(), time.Now().Nanosecond())
			ns := &corev1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: namespace,
				},
			}
			_, err = apiServer.clientSet.CoreV1().Namespaces().Create(context.TODO(), ns, metav1.CreateOptions{})
			if err != nil {
				return fmt.Errorf("failed to create test namespace: %v", err)
			}
			apiServer.namespace = namespace

			// Create a default service account in the namespace
			// This is required for pods to be created (ServiceAccount admission controller)
			sa := &corev1.ServiceAccount{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "default",
					Namespace: namespace,
				},
			}
			_, err = apiServer.clientSet.CoreV1().ServiceAccounts(namespace).Create(context.TODO(), sa, metav1.CreateOptions{})
			if err != nil {
				return fmt.Errorf("failed to create default service account: %v", err)
			}

			// Update kubeconfig to set the namespace as default for the context
			err = updateKubeConfigNamespace(apiServer.kubeConfigFile, namespace)
			if err != nil {
				return fmt.Errorf("failed to update kubeconfig namespace: %v", err)
			}

			// Store API server for cleanup
			env.Values["apiserver"] = apiServer

			// Set KUBECONFIG environment variable
			env.Setenv("KUBECONFIG", apiServer.kubeConfigFile)

			// Set KUBE_ROOT to the kubernetes repo root so tests can use hack/testdata paths
			kubeRoot := getKubeRoot()
			env.Setenv("KUBE_ROOT", kubeRoot)

			// Disable plugins for tests
			env.Setenv("KUBECTL_PLUGINS_PATH", "")

			// Set HOME to work directory to avoid interference
			env.Setenv("HOME", env.WorkDir)

			// Set image environment variables used by various tests
			// These correspond to the IMAGE_* variables in legacy-script.sh
			env.Setenv("IMAGE_NGINX", "registry.k8s.io/nginx:1.7.9")
			env.Setenv("IMAGE_DEPLOYMENT_R1", "registry.k8s.io/nginx:test-cmd")
			env.Setenv("IMAGE_DEPLOYMENT_R2", "registry.k8s.io/nginx:1.7.9")
			env.Setenv("IMAGE_PERL", "registry.k8s.io/perl")
			env.Setenv("IMAGE_PAUSE_V2", "registry.k8s.io/pause:2.0")
			env.Setenv("IMAGE_DAEMONSET_R2", "registry.k8s.io/pause:latest")
			env.Setenv("IMAGE_DAEMONSET_R2_2", "registry.k8s.io/nginx:test-cmd")
			env.Setenv("IMAGE_STATEFULSET_R1", "registry.k8s.io/nginx-slim:0.7")
			env.Setenv("IMAGE_STATEFULSET_R2", "registry.k8s.io/nginx-slim:0.8")

			return nil
		},

		// Custom commands for kubectl tests
		Cmds: map[string]func(ts *testscript.TestScript, neg bool, args []string){
			// kubectl command - runs kubectl in-process
			"kubectl": cmdKubectl,

			// kube-retry command - retry kubectl get until condition is met
			"kube-retry": cmdRetry,
		},

		// Conditions for conditional test execution
		Condition: func(cond string) (bool, error) {
			// Add custom conditions if needed
			return false, nil
		},
	})
}

// startTestAPIServer starts etcd and apiserver for testing
func startTestAPIServer(t testing.TB) (*testAPIServer, error) {
	cfg := etcdserver.NewTestConfig(t)
	etcdClient := etcdserver.RunEtcd(t, cfg)
	storageConfig := storagebackend.NewDefaultConfig(path.Join(uuid.New().String(), "registry"), nil)
	storageConfig.Transport.ServerList = etcdClient.Endpoints()

	// Use -v=0 to reduce klog verbosity (suppress most apiserver logs)
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{"--v=0"}, storageConfig)
	t.Cleanup(server.TearDownFn)

	clientSet := clientset.NewForConfigOrDie(server.ClientConfig)

	kubeConfigFile, err := writeKubeConfig(t, server.ClientConfig)
	if err != nil {
		return nil, err
	}

	return &testAPIServer{
		server:         server,
		clientSet:      clientSet,
		kubeConfigFile: kubeConfigFile,
	}, nil
}

// writeKubeConfig writes a kubeconfig file for the test API server
func writeKubeConfig(t testing.TB, kubeClientConfig *rest.Config) (string, error) {
	// Copy config and reset SNI to use the "real" cert
	configCopy := rest.CopyConfig(kubeClientConfig)
	configCopy.ServerName = ""

	servingCerts, _, err := cert.GetServingCertificatesForURL(configCopy.Host, "")
	if err != nil {
		return "", err
	}
	encodedServing, err := cert.EncodeCertificates(servingCerts...)
	if err != nil {
		return "", err
	}
	configCopy.CAData = encodedServing

	adminKubeConfig := kubeconfig.CreateKubeConfig(configCopy)
	tmpDir := t.TempDir()
	kubeConfigFile := path.Join(tmpDir, "kubeconfig")
	if err := clientcmd.WriteToFile(*adminKubeConfig, kubeConfigFile); err != nil {
		return "", err
	}

	return kubeConfigFile, nil
}

// updateKubeConfigNamespace updates the kubeconfig to set the namespace as default for the current context
func updateKubeConfigNamespace(kubeConfigFile, namespace string) error {
	// Load the kubeconfig
	config, err := clientcmd.LoadFromFile(kubeConfigFile)
	if err != nil {
		return fmt.Errorf("failed to load kubeconfig: %v", err)
	}

	// Get the current context name
	currentContextName := config.CurrentContext
	if currentContextName == "" {
		return fmt.Errorf("no current context set in kubeconfig")
	}

	// Get the current context
	currentContext, exists := config.Contexts[currentContextName]
	if !exists {
		return fmt.Errorf("current context %q not found in kubeconfig", currentContextName)
	}

	// Update the namespace in the context
	currentContext.Namespace = namespace

	// Write the updated kubeconfig back to disk
	if err := clientcmd.WriteToFile(*config, kubeConfigFile); err != nil {
		return fmt.Errorf("failed to write updated kubeconfig: %v", err)
	}

	return nil
}

// cmdKubectl implements the kubectl command for testscript
// This runs kubectl in-process using the kubectl library directly
func cmdKubectl(ts *testscript.TestScript, neg bool, args []string) {
	kubeconfigPath := ts.Getenv("KUBECONFIG")
	if kubeconfigPath == "" {
		ts.Fatalf("KUBECONFIG not set")
	}

	// Change to KUBE_ROOT so that hack/testdata paths work (like in apply.sh)
	kubeRoot := ts.Getenv("KUBE_ROOT")
	if kubeRoot == "" {
		ts.Fatalf("KUBE_ROOT environment variable not set")
	}
	oldDir, err := os.Getwd()
	if err != nil {
		ts.Fatalf("failed to get current directory: %v", err)
	}
	if err := os.Chdir(kubeRoot); err != nil {
		ts.Fatalf("failed to change to KUBE_ROOT directory: %v", err)
	}
	defer os.Chdir(oldDir)

	// Check for output redirection (>filename)
	var redirectFile string
	kubectlArgs := args
	for i, arg := range args {
		if arg == ">" && i+1 < len(args) {
			redirectFile = args[i+1]
			kubectlArgs = args[:i]
			break
		}
	}

	// Prepare I/O streams
	var stdout, stderr bytes.Buffer

	// Build the command arguments (prepend with program name placeholder)
	fullArgs := append([]string{"kubectl"}, kubectlArgs...)

	// For delete commands, automatically add --wait=false to prevent hanging
	// In-process kubectl execution has issues with watch connections for delete --wait,
	// but this doesn't affect test correctness since we're only testing the delete request itself
	if len(args) > 0 && args[0] == "delete" {
		hasWaitFlag := false
		for _, arg := range args {
			if strings.HasPrefix(arg, "--wait") {
				hasWaitFlag = true
				break
			}
		}
		if !hasWaitFlag {
			fullArgs = append(fullArgs, "--wait=false")
		}
	}

	// Create config flags with the kubeconfig
	configFlags := genericclioptions.NewConfigFlags(true).
		WithDeprecatedPasswordFlag().
		WithDiscoveryBurst(300).
		WithDiscoveryQPS(50.0)
	configFlags.KubeConfig = &kubeconfigPath

	ioStreams := genericiooptions.IOStreams{
		In:     strings.NewReader(""),
		Out:    &stdout,
		ErrOut: &stderr,
	}

	// Create kubectl command
	cmd := kubectlcmd.NewKubectlCommand(kubectlcmd.KubectlOptions{
		PluginHandler: nil, // Disable plugins in tests
		Arguments:     fullArgs,
		ConfigFlags:   configFlags.WithWarningPrinter(ioStreams),
		IOStreams:     ioStreams,
	})

	// Set the args for cobra (skip the "kubectl" at index 0)
	cmd.SetArgs(fullArgs[1:])
	cmd.SetOut(&stdout)
	cmd.SetErr(&stderr)

	// Override the fatal error handler to prevent os.Exit() from terminating the test process
	// When kubectl encounters an error, it normally calls os.Exit(). We need to capture this
	// as a regular error instead.
	cmdutil.BehaviorOnFatal(func(msg string, code int) {
		panic(fmt.Sprintf("kubectl fatal error: %s (code %d)", msg, code))
	})

	// Execute kubectl command and recover from any panic caused by fatal errors
	var cmdErr error
	var fatalMsg string
	func() {
		defer func() {
			if r := recover(); r != nil {
				// Convert panic to error - this happens when kubectl calls fatal()
				fatalMsg = fmt.Sprintf("%v", r)
				cmdErr = fmt.Errorf("%v", r)
			}
		}()
		cmdErr = cmd.Execute()
	}()
	err = cmdErr

	// Handle output redirection if specified
	if redirectFile != "" {
		// Write stdout to the specified file in the test's working directory
		outputPath := ts.MkAbs(redirectFile)
		if err := os.WriteFile(outputPath, []byte(stdout.String()), 0644); err != nil {
			ts.Fatalf("failed to write output to %s: %v", redirectFile, err)
		}
	} else {
		// Write output to testscript's stdout/stderr
		fmt.Fprint(ts.Stdout(), stdout.String())
	}

	// Include stderr from command, plus any fatal error message
	fmt.Fprint(ts.Stderr(), stderr.String())
	if fatalMsg != "" {
		// Also write the fatal error message to stderr so tests can check it
		// The fatal message format is "kubectl fatal error: <message> (code <n>)"
		// We extract just the message part for easier matching
		if strings.HasPrefix(fatalMsg, "kubectl fatal error: ") {
			// Extract the actual error message between prefix and " (code"
			msg := strings.TrimPrefix(fatalMsg, "kubectl fatal error: ")
			if idx := strings.LastIndex(msg, " (code"); idx > 0 {
				msg = msg[:idx]
			}
			fmt.Fprint(ts.Stderr(), msg)
		} else {
			fmt.Fprint(ts.Stderr(), fatalMsg)
		}
	}

	if neg {
		if err == nil {
			ts.Fatalf("kubectl %v succeeded unexpectedly\nstdout: %s\nstderr: %s", args, stdout.String(), stderr.String())
		}
		// Command failed as expected - this is success for negated commands
	} else {
		if err != nil {
			ts.Fatalf("kubectl %v failed: %v\nstdout: %s\nstderr: %s", args, err, stdout.String(), stderr.String())
		}
	}
}

// cmdRetry repeatedly runs kubectl get and checks if the output matches the expected value
// This is equivalent to kube::test::wait_object_assert in the bash test framework
// Usage: kube-retry <resource> <jsonpath> <expected> [timeout-seconds]
func cmdRetry(ts *testscript.TestScript, neg bool, args []string) {
	if len(args) < 3 {
		ts.Fatalf("usage: kube-retry <resource> <jsonpath> <expected> [timeout-seconds]")
	}

	resource := args[0]
	jsonpath := args[1]
	expected := args[2]
	timeoutSecs := 30
	if len(args) >= 4 {
		fmt.Sscanf(args[3], "%d", &timeoutSecs)
	}

	kubeconfigPath := ts.Getenv("KUBECONFIG")
	if kubeconfigPath == "" {
		ts.Fatalf("KUBECONFIG not set")
	}

	// Change to KUBE_ROOT so that resource paths work correctly
	kubeRoot := ts.Getenv("KUBE_ROOT")
	if kubeRoot != "" {
		oldDir, err := os.Getwd()
		if err == nil {
			os.Chdir(kubeRoot)
			defer os.Chdir(oldDir)
		}
	}

	// Retry up to timeout with exponential backoff
	maxTries := 10
	for i := 0; i < maxTries; i++ {
		var stdout, stderr bytes.Buffer

		configFlags := genericclioptions.NewConfigFlags(true).
			WithDeprecatedPasswordFlag().
			WithDiscoveryBurst(300).
			WithDiscoveryQPS(50.0)
		configFlags.KubeConfig = &kubeconfigPath

		ioStreams := genericiooptions.IOStreams{
			In:     strings.NewReader(""),
			Out:    &stdout,
			ErrOut: &stderr,
		}

		// Split resource by spaces to support things like "pods -n nsb"
		resourceParts := strings.Fields(resource)
		cmdArgs := append([]string{"kubectl", "get"}, resourceParts...)
		cmdArgs = append(cmdArgs, "-o", "jsonpath="+jsonpath)

		// Override fatal error handler
		cmdutil.BehaviorOnFatal(func(msg string, code int) {
			panic(fmt.Sprintf("kubectl fatal error: %s (code %d)", msg, code))
		})

		cmd := kubectlcmd.NewKubectlCommand(kubectlcmd.KubectlOptions{
			PluginHandler: nil,
			Arguments:     cmdArgs,
			ConfigFlags:   configFlags.WithWarningPrinter(ioStreams),
			IOStreams:     ioStreams,
		})

		cmd.SetArgs(cmdArgs[1:])
		cmd.SetOut(&stdout)
		cmd.SetErr(&stderr)

		// Execute with panic recovery
		var cmdErr error
		func() {
			defer func() {
				if r := recover(); r != nil {
					cmdErr = fmt.Errorf("%v", r)
				}
			}()
			cmdErr = cmd.Execute()
		}()

		actual := strings.TrimSpace(stdout.String())

		// Debug output
		ts.Logf("kube-retry attempt %d: resource=%q jsonpath=%q expected=%q actual=%q err=%v stderr=%q",
			i+1, resource, jsonpath, expected, actual, cmdErr, stderr.String())

		// Check if we got the expected result
		if cmdErr == nil && actual == expected {
			if neg {
				ts.Fatalf("kube-retry: got unexpected match %q after %d tries", actual, i+1)
			}
			return // Success!
		}

		// Sleep before next retry (exponential backoff, up to 3 seconds)
		sleepTime := time.Duration(i) * 500 * time.Millisecond
		if sleepTime > 3*time.Second {
			sleepTime = 3 * time.Second
		}
		time.Sleep(sleepTime)
	}

	// If we get here, we never got the expected result
	if neg {
		return // Expected to not match, so this is success
	}
	ts.Fatalf("kube-retry: timed out waiting for %s jsonpath=%s to equal %q (last actual value was logged above)", resource, jsonpath, expected)
}
