// +build linux

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

// To run tests in this suite
// NOTE: This test suite requires password-less sudo capabilities to run the kubelet and kube-apiserver.
package e2enode

import (
	"flag"
	"math/rand"
	"os"
	"testing"
	"time"

	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/kubernetes/test/e2e/framework"
	e2econfig "k8s.io/kubernetes/test/e2e/framework/config"
	e2etestfiles "k8s.io/kubernetes/test/e2e/framework/testfiles"
	"k8s.io/kubernetes/test/e2e/generated"

	"github.com/spf13/pflag"
)

// registerNodeFlags registers flags specific to the node e2e test suite.
func registerNodeFlags(flags *flag.FlagSet) {
	// Mark the test as node e2e when node flags are api.Registry.
	framework.TestContext.NodeE2E = true
	flags.StringVar(&framework.TestContext.BearerToken, "bearer-token", "", "The bearer token to authenticate with. If not specified, it would be a random token. Currently this token is only used in node e2e tests.")
	flags.StringVar(&framework.TestContext.NodeName, "node-name", "", "Name of the node to run tests on.")
	// TODO(random-liu): Move kubelet start logic out of the test.
	// TODO(random-liu): Move log fetch logic out of the test.
	// There are different ways to start kubelet (systemd, initd, docker, manually started etc.)
	// and manage logs (journald, upstart etc.).
	// For different situation we need to mount different things into the container, run different commands.
	// It is hard and unnecessary to deal with the complexity inside the test suite.
	flags.BoolVar(&framework.TestContext.NodeConformance, "conformance", false, "If true, the test suite will not start kubelet, and fetch system log (kernel, docker, kubelet log etc.) to the report directory.")
	flags.BoolVar(&framework.TestContext.PrepullImages, "prepull-images", true, "If true, prepull images so image pull failures do not cause test failures.")
	flags.StringVar(&framework.TestContext.ImageDescription, "image-description", "", "The description of the image which the test will be running on.")
	flags.StringVar(&framework.TestContext.SystemSpecName, "system-spec-name", "", "The name of the system spec (e.g., gke) that's used in the node e2e test. The system specs are in test/e2e_node/system/specs/. This is used by the test framework to determine which tests to run for validating the system requirements.")
	flags.Var(cliflag.NewMapStringString(&framework.TestContext.ExtraEnvs), "extra-envs", "The extra environment variables needed for node e2e tests. Format: a list of key=value pairs, e.g., env1=val1,env2=val2")
	flags.StringVar(&framework.TestContext.SriovdpConfigMapFile, "sriovdp-configmap-file", "", "The name of the SRIOV device plugin Config Map to load.")
	flag.StringVar(&framework.TestContext.ClusterDNSDomain, "dns-domain", "", "The DNS Domain of the cluster.")
}

func init() {
	// Enable bindata file lookup as fallback.
	e2etestfiles.AddFileSource(e2etestfiles.BindataFileSource{
		Asset:      generated.Asset,
		AssetNames: generated.AssetNames,
	})

}

func TestMain(m *testing.M) {
	// Copy go flags in TestMain, to ensure go test flags are registered (no longer available in init() as of go1.13)
	e2econfig.CopyFlags(e2econfig.Flags, flag.CommandLine)
	framework.RegisterCommonFlags(flag.CommandLine)
	registerNodeFlags(flag.CommandLine)
	pflag.CommandLine.AddGoFlagSet(flag.CommandLine)
	// Mark the run-services-mode flag as hidden to prevent user from using it.
	pflag.CommandLine.MarkHidden("run-services-mode")
	// It's weird that if I directly use pflag in TestContext, it will report error.
	// It seems that someone is using flag.Parse() after init() and TestMain().
	// TODO(random-liu): Find who is using flag.Parse() and cause errors and move the following logic
	// into TestContext.
	// TODO(pohly): remove RegisterNodeFlags from test_context.go enable Viper config support here?

	rand.Seed(time.Now().UnixNano())
	pflag.Parse()
	framework.AfterReadingAllFlags(&framework.TestContext)
	setExtraEnvs()
	os.Exit(m.Run())
}

func TestE2eNode(t *testing.T) {
	RunE2ENodeTests(t)
}
