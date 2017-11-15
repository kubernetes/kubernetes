/*
Copyright 2016 The Kubernetes Authors.

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

package services

import (
	"flag"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/golang/glog"
	"github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/cmd/kubelet/app/options"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/scheme"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/v1alpha1"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e_node/builder"
)

// TODO(random-liu): Replace this with standard kubelet launcher.

// args is the type used to accumulate args from the flags with the same name.
type args []string

// String function of flag.Value
func (a *args) String() string {
	return fmt.Sprint(*a)
}

// Set function of flag.Value
func (a *args) Set(value string) error {
	// Someone else is calling flag.Parse after the flags are parsed in the
	// test framework. Use this to avoid the flag being parsed twice.
	// TODO(random-liu): Figure out who is parsing the flags.
	if flag.Parsed() {
		return nil
	}
	// Note that we assume all white space in flag string is separating fields
	na := strings.Fields(value)
	*a = append(*a, na...)
	return nil
}

// kubeletArgs is the override kubelet args specified by the test runner.
var kubeletArgs args

func init() {
	flag.Var(&kubeletArgs, "kubelet-flags", "Kubelet flags passed to kubelet, this will override default kubelet flags in the test. Flags specified in multiple kubelet-flags will be concatenate.")
}

// RunKubelet starts kubelet and waits for termination signal. Once receives the
// termination signal, it will stop the kubelet gracefully.
func RunKubelet() {
	var err error
	// Enable monitorParent to make sure kubelet will receive termination signal
	// when test process exits.
	e := NewE2EServices(true /* monitorParent */)
	defer e.Stop()
	e.kubelet, err = e.startKubelet()
	if err != nil {
		glog.Fatalf("Failed to start kubelet: %v", err)
	}
	// Wait until receiving a termination signal.
	waitForTerminationSignal()
}

const (
	// Ports of different e2e services.
	kubeletPort          = "10250"
	kubeletReadOnlyPort  = "10255"
	KubeletRootDirectory = "/var/lib/kubelet"
	// Health check url of kubelet
	kubeletHealthCheckURL = "http://127.0.0.1:" + kubeletReadOnlyPort + "/healthz"
)

// startKubelet starts the Kubelet in a separate process or returns an error
// if the Kubelet fails to start.
func (e *E2EServices) startKubelet() (*server, error) {
	glog.Info("Starting kubelet")

	// set feature gates so we can check which features are enabled and pass the appropriate flags
	utilfeature.DefaultFeatureGate.Set(framework.TestContext.FeatureGates)

	// Build kubeconfig
	kubeconfigPath, err := createKubeconfigCWD()
	if err != nil {
		return nil, err
	}

	// Create pod manifest path
	manifestPath, err := createPodManifestDirectory()
	if err != nil {
		return nil, err
	}
	e.rmDirs = append(e.rmDirs, manifestPath)
	err = createRootDirectory(KubeletRootDirectory)
	if err != nil {
		return nil, err
	}
	var killCommand, restartCommand *exec.Cmd
	var isSystemd bool
	// Apply default kubelet flags.
	cmdArgs := []string{}
	kubeArgs := []string{}
	if systemdRun, err := exec.LookPath("systemd-run"); err == nil {
		// On systemd services, detection of a service / unit works reliably while
		// detection of a process started from an ssh session does not work.
		// Since kubelet will typically be run as a service it also makes more
		// sense to test it that way
		isSystemd = true
		unitName := fmt.Sprintf("kubelet-%d.service", rand.Int31())
		cmdArgs = append(cmdArgs, systemdRun, "--unit="+unitName, "--slice=runtime.slice", "--remain-after-exit", builder.GetKubeletServerBin())
		killCommand = exec.Command("systemctl", "kill", unitName)
		restartCommand = exec.Command("systemctl", "restart", unitName)
		e.logs["kubelet.log"] = LogFileData{
			Name:              "kubelet.log",
			JournalctlCommand: []string{"-u", unitName},
		}
		kubeArgs = append(kubeArgs,
			"--kubelet-cgroups=/kubelet.slice",
			"--cgroup-root=/",
		)
	} else {
		cmdArgs = append(cmdArgs, builder.GetKubeletServerBin())
		kubeArgs = append(kubeArgs,
			// TODO(random-liu): Get rid of this docker specific thing.
			"--runtime-cgroups=/docker-daemon",
			"--kubelet-cgroups=/kubelet",
			"--cgroup-root=/",
			"--system-cgroups=/system",
		)
	}
	kubeArgs = append(kubeArgs,
		"--kubeconfig", kubeconfigPath,
		"--address", "0.0.0.0",
		"--port", kubeletPort,
		"--read-only-port", kubeletReadOnlyPort,
		"--root-dir", KubeletRootDirectory,
		"--volume-stats-agg-period", "10s", // Aggregate volumes frequently so tests don't need to wait as long
		"--allow-privileged", "true",
		"--serialize-image-pulls", "false",
		"--pod-manifest-path", manifestPath,
		"--file-check-frequency", "10s", // Check file frequently so tests won't wait too long
		"--docker-disable-shared-pid=false",
		// Assign a fixed CIDR to the node because there is no node controller.
		//
		// Note: this MUST be in sync with with the IP in
		// - cluster/gce/config-test.sh and
		// - test/e2e_node/conformance/run_test.sh.
		"--pod-cidr", "10.100.0.0/24",
		"--eviction-pressure-transition-period", "30s",
		// Apply test framework feature gates by default. This could also be overridden
		// by kubelet-flags.
		"--feature-gates", framework.TestContext.FeatureGates,
		"--eviction-hard", "memory.available<250Mi,nodefs.available<10%,nodefs.inodesFree<5%", // The hard eviction thresholds.
		"--eviction-minimum-reclaim", "nodefs.available=5%,nodefs.inodesFree=5%", // The minimum reclaimed resources after eviction.
		"--v", LOG_VERBOSITY_LEVEL, "--logtostderr",
	)

	if utilfeature.DefaultFeatureGate.Enabled(features.DynamicKubeletConfig) {
		// Enable dynamic config if the feature gate is enabled
		dir, err := getDynamicConfigDirectory()
		if err != nil {
			return nil, err
		}
		kubeArgs = append(kubeArgs, "--dynamic-config-dir", dir)
	}

	// Enable kubenet by default.
	cniBinDir, err := getCNIBinDirectory()
	if err != nil {
		return nil, err
	}

	cniConfDir, err := getCNIConfDirectory()
	if err != nil {
		return nil, err
	}

	kubeArgs = append(kubeArgs,
		"--network-plugin=kubenet",
		"--cni-bin-dir", cniBinDir,
		"--cni-conf-dir", cniConfDir)

	// Keep hostname override for convenience.
	if framework.TestContext.NodeName != "" { // If node name is specified, set hostname override.
		kubeArgs = append(kubeArgs, "--hostname-override", framework.TestContext.NodeName)
	}

	// Override the default kubelet flags.
	kubeArgs = append(kubeArgs, kubeletArgs...)

	// If the config file feature gate is enabled, generate the file and remove the flags it applies to
	if utilfeature.DefaultFeatureGate.Enabled(features.KubeletConfigFile) {
		kc, other, err := splitKubeletConfigArgs(kubeArgs)
		if err != nil {
			return nil, err
		}
		// replace kubeArgs with the new command line, which has had the KubeletConfiguration flags removed
		kubeArgs = other
		path, err := writeKubeletConfigFile(kc)
		if err != nil {
			return nil, err
		}
		// ensure the test context feature gates (typically DynamicKubeletConfig and KubeletConfigFile)
		// are set on the command line
		kubeArgs = append(kubeArgs, "--feature-gates", framework.TestContext.FeatureGates)
		// add the flag to load config from a file
		kubeArgs = append(kubeArgs, "--init-config-dir", filepath.Dir(path))
	}

	// combine the kubelet parameters with the command
	cmdArgs = append(cmdArgs, kubeArgs...)

	// Adjust the args if we are running kubelet with systemd.
	if isSystemd {
		adjustArgsForSystemd(cmdArgs)
	}

	cmd := exec.Command(cmdArgs[0], cmdArgs[1:]...)
	server := newServer(
		"kubelet",
		cmd,
		killCommand,
		restartCommand,
		[]string{kubeletHealthCheckURL},
		"kubelet.log",
		e.monitorParent,
		true /* restartOnExit */)
	return server, server.start()
}

// splitKubeletConfigArgs parses args onto a KubeletConfiguration object and also returns the unknown args
func splitKubeletConfigArgs(args []string) (*kubeletconfig.KubeletConfiguration, []string, error) {
	kc, err := options.NewKubeletConfiguration()
	if err != nil {
		return nil, nil, err
	}
	fs := pflag.NewFlagSet("kubeletconfig", pflag.ContinueOnError)
	options.AddKubeletConfigFlags(fs, kc)
	known, other := splitKnownArgs(fs, args)
	if err := fs.Parse(known); err != nil {
		return nil, nil, err
	}
	return kc, other, nil
}

// splitKnownArgs splits argument list into those known by the flagset, and those not known
// only tests for longhand args, e.g. prefixed with `--`
// TODO(mtaufen): I don't think the kubelet has any shorthand args, but if it does we will need to modify this.
func splitKnownArgs(fs *pflag.FlagSet, args []string) ([]string, []string) {
	known := []string{}
	other := []string{}
	lastFlag := len(args)
	for i := len(args) - 1; i >= 0; i-- {
		if strings.HasPrefix(args[i], "--") {
			if fs.Lookup(strings.TrimPrefix(args[i], "--")) == nil {
				// flag is unknown, add flag and params to other
				// prepend to maintain order
				other = append(append([]string(nil), args[i:lastFlag]...), other...)
				// cut from known
			} else {
				// flag is known, add flag and params to known
				// prepend to maintain order
				known = append(append([]string(nil), args[i:lastFlag]...), known...)
			}
			// mark the last location where we saw a flag
			lastFlag = i
		}
	}
	return known, other
}

// writeKubeletConfigFile writes the kubelet config file based on the args and returns the filename
func writeKubeletConfigFile(internal *kubeletconfig.KubeletConfiguration) (string, error) {
	// extract the KubeletConfiguration and convert to versioned
	versioned := &v1alpha1.KubeletConfiguration{}
	scheme, _, err := scheme.NewSchemeAndCodecs()
	if err != nil {
		return "", err
	}
	if err := scheme.Convert(internal, versioned, nil); err != nil {
		return "", err
	}
	// encode
	encoder, err := newKubeletConfigJSONEncoder()
	if err != nil {
		return "", err
	}
	data, err := runtime.Encode(encoder, versioned)
	if err != nil {
		return "", err
	}
	// create the init conifg directory
	dir, err := createKubeletInitConfigDirectory()
	if err != nil {
		return "", err
	}
	// write init config file
	path := filepath.Join(dir, "kubelet")
	if err := ioutil.WriteFile(path, data, 0755); err != nil {
		return "", err
	}
	return path, nil
}

func newKubeletConfigJSONEncoder() (runtime.Encoder, error) {
	_, kubeletCodecs, err := scheme.NewSchemeAndCodecs()
	if err != nil {
		return nil, err
	}

	mediaType := "application/json"
	info, ok := runtime.SerializerInfoForMediaType(kubeletCodecs.SupportedMediaTypes(), mediaType)
	if !ok {
		return nil, fmt.Errorf("unsupported media type %q", mediaType)
	}
	return kubeletCodecs.EncoderForVersion(info.Serializer, v1alpha1.SchemeGroupVersion), nil
}

// createPodManifestDirectory creates pod manifest directory.
func createPodManifestDirectory() (string, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("failed to get current working directory: %v", err)
	}
	path, err := ioutil.TempDir(cwd, "pod-manifest")
	if err != nil {
		return "", fmt.Errorf("failed to create static pod manifest directory: %v", err)
	}
	return path, nil
}

// createKubeconfig creates a kubeconfig file at the fully qualified `path`. The parent dirs must exist.
func createKubeconfig(path string) error {
	kubeconfig := []byte(`apiVersion: v1
kind: Config
users:
- name: kubelet
clusters:
- cluster:
    server: ` + getAPIServerClientURL() + `
    insecure-skip-tls-verify: true
  name: local
contexts:
- context:
    cluster: local
    user: kubelet
  name: local-context
current-context: local-context`)

	if err := ioutil.WriteFile(path, kubeconfig, 0666); err != nil {
		return err
	}
	return nil
}

func createRootDirectory(path string) error {
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			return os.MkdirAll(path, os.FileMode(0755))
		} else {
			return err
		}
	}
	return nil
}

func kubeconfigCWDPath() (string, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("failed to get current working directory: %v", err)
	}
	return filepath.Join(cwd, "kubeconfig"), nil
}

// like createKubeconfig, but creates kubeconfig at current-working-directory/kubeconfig
// returns a fully-qualified path to the kubeconfig file
func createKubeconfigCWD() (string, error) {
	kubeconfigPath, err := kubeconfigCWDPath()
	if err != nil {
		return "", err
	}

	if err = createKubeconfig(kubeconfigPath); err != nil {
		return "", err
	}
	return kubeconfigPath, nil
}

// getCNIBinDirectory returns CNI directory.
func getCNIBinDirectory() (string, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	return filepath.Join(cwd, "cni", "bin"), nil
}

// getCNIConfDirectory returns CNI Configuration directory.
func getCNIConfDirectory() (string, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	return filepath.Join(cwd, "cni", "net.d"), nil
}

// getDynamicConfigDir returns the directory for dynamic Kubelet configuration
func getDynamicConfigDirectory() (string, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	return filepath.Join(cwd, "dynamic-kubelet-config"), nil
}

// createKubeletInitConfigDirectory creates and returns the name of the directory for dynamic Kubelet configuration
func createKubeletInitConfigDirectory() (string, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	path := filepath.Join(cwd, "init-kubelet-config")
	if err := os.MkdirAll(path, 0755); err != nil {
		return "", err
	}
	return path, nil
}

// adjustArgsForSystemd escape special characters in kubelet arguments for systemd. Systemd
// may try to do auto expansion without escaping.
func adjustArgsForSystemd(args []string) {
	for i := range args {
		args[i] = strings.Replace(args[i], "%", "%%", -1)
		args[i] = strings.Replace(args[i], "$", "$$", -1)
	}
}
