/*
Copyright 2021 The Kubernetes Authors.

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

// Package node implements `e2e_node.test remote <flags>`, i.e.
// it gets built into e2e_node.test. This avoids shipping another binary
// in release packages. Only remote execution of the E2E suite is
// supported.
//
// It gets called by `kubetest2 noop -test=node -- <flags>`
// when the flags include any of the "use or build binaries" variants.
// This enables running E2E node tests without the Kubernetes source code.
//
// It provides the command line flags of the kubetest2 node tester (with some
// changes, see below) and maps them to the way how `make test-e2e-node` would
// have invoked test/e2e_node/runner/remote.
//
// It's derived from
// https://github.com/kubernetes-sigs/kubetest2/blob/master/pkg/testers/node/node.go
// (revision 558f16b589d15d031595af7d035330b8e87bcaaf).
package node

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"time"

	"k8s.io/klog/v2"

	"k8s.io/kubernetes/test/e2e_node/remote"
	_ "k8s.io/kubernetes/test/e2e_node/remote/gce" // Register remote execution via GCE.
)

// Tester is intentionally as similar as possible to
// https://github.com/kubernetes-sigs/kubetest2/blob/master/pkg/testers/node/node.go
// to make it more obvious that the command line is the same,
// with a few flags added for binary paths.
type Tester struct {
	// Flags passed through by node tester.
	GCPProject       string        `desc:"GCP Project to create VMs in."`
	GCPZone          string        `desc:"GCP Zone to create VMs in."`
	SkipRegex        string        `desc:"Regular expression of jobs to skip."`
	FocusRegex       string        `desc:"Regular expression of jobs to focus on."`
	TestArgs         string        `desc:"A space-separated list of arguments to pass to node e2e test."`
	LabelFilter      string        `desc:"Label filter arguments to be passed to ginkgo."`
	ImageConfigFile  string        `desc:"Path to a file containing image configuration."`
	Images           string        `desc:"List of images to use when creating instances separated by commas"`
	ImageProject     string        `desc:"A GCP Project containing an image to use when creating instances"`
	InstanceType     string        `desc:"Machine/Instance type to use on AWS/GCP"`
	InstanceMetadata string        `desc:"Instance Metadata to use for creating GCE instance"`
	UserDataFile     string        `desc:"User Data to use for creating EC2 instance"`
	Provider         string        `desc:"Cloud Provider to use for node tests. Valid options are ec2 and gce"`
	ImageConfigDir   string        `desc:"Path to image config files."`
	Parallelism      int           `desc:"The number of nodes to run in parallel."`
	RuntimeConfig    string        `desc:"The runtime configuration for the API server. Format: a list of key=value pairs."`
	Timeout          time.Duration `desc:"How long (in golang duration format) to wait for ginkgo tests to complete."`
	DeleteInstances  bool          `desc:"Where to delete instances after running the test"`
	NodeEnv          string        `desc:"Additional metadata keys to add to a gce instance"`

	// Flags for pre-built binary support. Required, building from source is not supported.
	GinkgoBinary  string `desc:"Existing Ginkgo binary to be used on the target instead of building from source"`
	KubeletBinary string `desc:"Existing kubelet binary to be used on the target instead of building from source"`
	E2ENodeBinary string `desc:"Existing e2e-node.test binary to be used on the target instead of building from source"`

	// Flags for values determined by node tester.
	SSHUser string `desc:"Used for ssh into the remote node."`
	SSHKey  string `desc:"Used for ssh into the remote node."`
}

func NewDefaultTester() *Tester {
	return &Tester{
		Parallelism:     8,
		DeleteInstances: true,
	}
}

func (t *Tester) Execute(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet(`"e2e_node.test remote"`, flag.ExitOnError)

	// Bind all exported fields to flags. This replaces github.com/octago/sflags/gen/gpflag.
	value := reflect.ValueOf(t).Elem()
	for field := range reflect.TypeOf(*t).Fields() {
		if !field.IsExported() {
			continue
		}
		name := fieldToFlagName(field.Name)
		desc := field.Tag.Get("desc")
		ptr := value.FieldByIndex(field.Index).Addr()
		switch ptr := ptr.Interface().(type) {
		case *string:
			fs.StringVar(ptr, name, *ptr, desc)
		case *int:
			fs.IntVar(ptr, name, *ptr, desc)
		case *bool:
			fs.BoolVar(ptr, name, *ptr, desc)
		case *time.Duration:
			fs.DurationVar(ptr, name, *ptr, desc)
		default:
			return fmt.Errorf("unsupported config field type %T", ptr)
		}
	}

	klog.InitFlags(fs)
	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("parse flags: %v", err)
	}
	if err := t.validateFlags(); err != nil {
		return fmt.Errorf("validate flags: %v", err)
	}

	return t.Test()
}

// fieldToFlagName converts to lower case and inserts hyphens between the boundary
// between upper and lower characters. As a special case, e.g. "GCPProject" becomes
// "gcp-project".
func fieldToFlagName(in string) string {
	parts := regexp.MustCompile(`[a-z0-9]+|[A-Z0-9]+`).FindAllString(in, -1)
	for i, part := range parts {
		if part[0] < 'A' || part[0] > 'Z' {
			continue
		}
		if l := len(part); l > 1 {
			// Split before the last upper character.
			part = part[:l-1] + "-" + part[l-1:]
		}
		if i > 0 {
			part = "-" + part
		}
		parts[i] = strings.ToLower(part)
	}
	return strings.Join(parts, "")
}

func (t *Tester) validateFlags() error {
	if t.GinkgoBinary == "" {
		return errors.New("required --ginkgo-binary path missing")
	}
	if t.KubeletBinary == "" {
		return errors.New("required --kubelet-binary path missing")
	}
	if t.E2ENodeBinary == "" {
		return errors.New("required --e2e-node-binary path missing")
	}
	if t.GCPZone == "" && t.Provider == "gce" {
		return errors.New("required --gcp-zone")
	}
	return nil
}

// configureRemote configures the "remote" package through it's flags.
// This corresponds to test/e2e_node/runner/remote/run_remote.go
// as invoked by hack/make-rules/test-e2e-node.sh.
func (t *Tester) configureRemote() (finalErr error) {
	for name, value := range map[string]string{
		"ssh-env": "gce", // Hard-coded as in https://github.com/kubernetes/kubernetes/blob/34341909b3e9a4854ab5d336b056b934bbbd9f16/hack/make-rules/test-e2e-node.sh#L218

		// See test/e2e_node/remote/run_remote_suite.go and test/e2e_node/remote/gce/gce_runner.go
		// for the definition of these flags or check `go run ./test/e2e_node/runner/remote/ -help`.
		"project":           t.GCPProject,
		"zone":              t.GCPZone,
		"test_args":         t.TestArgs,
		"node-env":          t.NodeEnv,
		"delete-instances":  strconv.FormatBool(t.DeleteInstances),
		"image-config-file": t.ImageConfigFile,
		"image-config-dir":  t.ImageConfigDir,
		"image-project":     t.ImageProject,
		"images":            t.Images,
		"instance-metadata": t.InstanceMetadata,
		"instance-type":     t.InstanceType,
		"test-timeout":      t.Timeout.String(),

		"ginkgo-binary":   t.GinkgoBinary,
		"kubelet-binary":  t.KubeletBinary,
		"e2e-node-binary": t.E2ENodeBinary,

		"ssh-user": t.SSHUser,
		"ssh-key":  t.SSHKey,

		// Not used by the remote runner and cannot be set because the flag is only defined in the command:
		// https://github.com/kubernetes/kubernetes/blob/a5098cf9a1405b6b6ed6cf9e0e4e49270c5a0996/test/e2e_node/runner/remote/run_remote.go#L37-L41
		//
		// We accept the parameter for the sake of consistency and because it might make
		// migrating jobs easier. Maybe it will also be supported in the future, if there
		// turns out to be a need for it.
		// "runtime-config":    t.RuntimeConfig,

		// Combining multiple different parameters inside a single parameter is problematic
		// because of quoting, but that is what the existing code uses. A better solution
		// would be have a `ginkgo-flag` parameter that can be used once for each individual
		// Ginkgo parameter.
		"ginkgo-flags": fmt.Sprintf("--nodes=%d --label-filter=%q --skip=%q --focus=%q", t.Parallelism, t.LabelFilter, t.SkipRegex, t.FocusRegex),
	} {
		if err := remote.CommandLine.Set(name, value); err != nil {
			return fmt.Errorf("set --%s to %q: %v", name, value, err)
		}
	}

	return nil
}

func (t *Tester) Test() error {
	if err := t.configureRemote(); err != nil {
		return fmt.Errorf("configure remote E2E node execution: %v", err)
	}

	suite, err := remote.GetTestSuite("default")
	if err != nil {
		return fmt.Errorf("error looking up testsuite [%v] - registered test suites [%v]", err, remote.GetTestSuiteKeys())
	}
	remote.RunRemoteTestSuite(suite)
	return nil
}

func Main(args []string) {
	// No cancellation, there's nothing to clean up when killed.
	ctx := context.Background()

	t := NewDefaultTester()
	if err := t.Execute(ctx, args); err != nil {
		fmt.Fprintf(os.Stderr, "\"%s remote\"  failed: %v\n", os.Args[0], err)
		os.Exit(1)
	}
}
