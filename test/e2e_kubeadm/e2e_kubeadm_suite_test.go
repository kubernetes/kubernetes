/*
Copyright 2018 The Kubernetes Authors.

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

package e2e_kubeadm

import (
	"flag"
	"os"
	"testing"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
	"github.com/spf13/pflag"

	"k8s.io/kubernetes/test/e2e/framework"
)

func init() {
	framework.RegisterCommonFlags()
	framework.RegisterClusterFlags()

	pflag.CommandLine.AddGoFlagSet(flag.CommandLine)
}

func TestMain(m *testing.M) {
	pflag.Parse()
	framework.AfterReadingAllFlags(&framework.TestContext)
	os.Exit(m.Run())
}

func TestE2E(t *testing.T) {
	reporters := []ginkgo.Reporter{}
	gomega.RegisterFailHandler(ginkgo.Fail)
	ginkgo.RunSpecsWithDefaultAndCustomReporters(t, "E2EKubadm suite", reporters)
}
