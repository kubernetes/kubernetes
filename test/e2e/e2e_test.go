/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package e2e

import (
	"flag"
	"fmt"
	"path"
	"strings"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
	"github.com/onsi/ginkgo"
	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/reporters"
	"github.com/onsi/ginkgo/types"
	"github.com/onsi/gomega"
)

const (
	// podStartupTimeout is the time to allow all pods in the cluster to become
	// running and ready before any e2e tests run. It includes pulling all of
	// the pods (as of 5/18/15 this is 8 pods).
	podStartupTimeout = 10 * time.Minute

	// minStartupPods is the minimum number of pods that will allow
	// waitForPodsRunningReady(...) to succeed (i.e. WLOG if you know that
	// "DNS", and "Prometheus" pods need to be running, you might set it to "2").
	// More verbosely, that function
	// checks that all pods in the cluster are both in a phase of "running" and
	// have a condition of "ready": "true". It aims to ensure that the cluster's
	// pods are fully healthy before beginning e2e tests. However, if there were
	// only 0 pods, it would technically pass if there wasn't a required minimum
	// number of pods. We expect every cluster to come up with some number of
	// pods (which in practice is more than this number), so we have this
	// minimum here as a sanity check to make sure that there are actually pods
	// on the cluster (i.e. preventing a possible race with kube-addons). This
	// does *not* mean that the function will succeed as soon as minStartupPods
	// are found to be running and ready; it ensures that *all* pods it finds
	// are running and ready. This is the minimum number it must find.
	// TODO : Add command line option for this so that the number is non trivial.
	minStartupPods = 0
)

var (
	cloudConfig = &testContext.CloudConfig

	reportDir = flag.String("report-dir", "", "Path to the directory where the JUnit XML reports should be saved. Default is empty, which doesn't generate these reports.")
)

type failReporter struct {
	failed bool
}

func (f *failReporter) SpecSuiteWillBegin(config config.GinkgoConfigType, summary *types.SuiteSummary) {
}
func (f *failReporter) BeforeSuiteDidRun(setupSummary *types.SetupSummary) {}
func (f *failReporter) SpecWillRun(specSummary *types.SpecSummary)         {}
func (f *failReporter) SpecDidComplete(specSummary *types.SpecSummary) {
	if specSummary.Failed() {
		f.failed = true
	}
}
func (f *failReporter) AfterSuiteDidRun(setupSummary *types.SetupSummary) {}
func (f *failReporter) SpecSuiteDidEnd(summary *types.SuiteSummary)       {}

func init() {
	// Turn on verbose by default to get spec names
	config.DefaultReporterConfig.Verbose = true

	// Turn on EmitSpecProgress to get spec progress (especially on interrupt)
	config.GinkgoConfig.EmitSpecProgress = true

	// Randomize specs as well as suites
	config.GinkgoConfig.RandomizeAllSpecs = true

	flag.StringVar(&testContext.KubeConfig, clientcmd.RecommendedConfigPathFlag, "", "Path to kubeconfig containing embeded authinfo.")
	flag.StringVar(&testContext.KubeContext, clientcmd.FlagContext, "", "kubeconfig context to use/override. If unset, will use value from 'current-context'")
	flag.StringVar(&testContext.CertDir, "cert-dir", "", "Path to the directory containing the certs. Default is empty, which doesn't use certs.")
	flag.StringVar(&testContext.Host, "host", "", "The host, or apiserver, to connect to")
	flag.StringVar(&testContext.RepoRoot, "repo-root", "../../", "Root directory of kubernetes repository, for finding test files.")
	flag.StringVar(&testContext.Provider, "provider", "", "The name of the Kubernetes provider (gce, gke, local, vagrant, etc.)")
	flag.StringVar(&testContext.KubectlPath, "kubectl-path", "kubectl", "The kubectl binary to use. For development, you might use 'cluster/kubectl.sh' here.")

	// TODO: Flags per provider?  Rename gce-project/gce-zone?
	flag.StringVar(&cloudConfig.MasterName, "kube-master", "", "Name of the kubernetes master. Only required if provider is gce or gke")
	flag.StringVar(&cloudConfig.ProjectID, "gce-project", "", "The GCE project being used, if applicable")
	flag.StringVar(&cloudConfig.Zone, "gce-zone", "", "GCE zone being used, if applicable")
	flag.StringVar(&cloudConfig.NodeInstanceGroup, "node-instance-group", "", "Name of the managed instance group for nodes. Valid only for gce")
	flag.IntVar(&cloudConfig.NumNodes, "num-nodes", -1, "Number of nodes in the cluster")

	flag.StringVar(&cloudConfig.ClusterTag, "cluster-tag", "", "Tag used to identify resources.  Only required if provider is aws.")
}

func TestE2E(t *testing.T) {
	util.ReallyCrash = true
	util.InitLogs()
	defer util.FlushLogs()

	// TODO: possibly clean up or refactor this functionality.
	if testContext.Provider == "" {
		glog.Fatal("The --provider flag is not set.  Treating as a conformance test.  Some tests may not be run.")
	}

	if testContext.Provider == "aws" {
		awsConfig := "[Global]\n"
		if cloudConfig.Zone == "" {
			glog.Fatal("gce-zone must be specified for AWS")
		}
		awsConfig += fmt.Sprintf("Zone=%s\n", cloudConfig.Zone)

		if cloudConfig.ClusterTag == "" {
			glog.Fatal("--cluster-tag must be specified for AWS")
		}
		awsConfig += fmt.Sprintf("KubernetesClusterTag=%s\n", cloudConfig.ClusterTag)

		var err error
		cloudConfig.Provider, err = cloudprovider.GetCloudProvider(testContext.Provider, strings.NewReader(awsConfig))
		if err != nil {
			glog.Fatal("Error building AWS provider: ", err)
		}
	}

	// Disable density test unless it's explicitly requested.
	if config.GinkgoConfig.FocusString == "" && config.GinkgoConfig.SkipString == "" {
		config.GinkgoConfig.SkipString = "Skipped"
	}
	gomega.RegisterFailHandler(ginkgo.Fail)

	// Ensure all pods are running and ready before starting tests (otherwise,
	// cluster infrastructure pods that are being pulled or started can block
	// test pods from running, and tests that ensure all pods are running and
	// ready will fail).
	if err := waitForPodsRunningReady(api.NamespaceDefault, minStartupPods, podStartupTimeout); err != nil {
		glog.Fatalf("Error waiting for all pods to be running and ready: %v", err)
	}

	// Run tests through the Ginkgo runner with output to console + JUnit for Jenkins
	var r []ginkgo.Reporter
	if *reportDir != "" {
		r = append(r, reporters.NewJUnitReporter(path.Join(*reportDir, fmt.Sprintf("junit_%02d.xml", config.GinkgoConfig.ParallelNode))))
		failReport := &failReporter{}
		r = append(r, failReport)
		defer func() {
			if failReport.failed {
				coreDump(*reportDir)
			}
		}()
	}
	ginkgo.RunSpecsWithDefaultAndCustomReporters(t, "Kubernetes e2e suite", r)
}
