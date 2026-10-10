/*
Copyright 2023 The Kubernetes Authors.

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

package app

import (
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/cloud-provider/names"
	"k8s.io/cloud-provider/options"
	cliflag "k8s.io/component-base/cli/flag"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	logsapi "k8s.io/component-base/logs/api/v1"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/features"
)

func TestCloudControllerNamesConsistency(t *testing.T) {
	controllerNameRegexp := regexp.MustCompile("^[a-z]([-a-z]*[a-z])?$")

	for name := range DefaultInitFuncConstructors {
		if !controllerNameRegexp.MatchString(name) {
			t.Errorf("name consistency check failed: controller %q must consist of lower case alphabetic characters or '-', and must start and end with an alphabetic character", name)
		}
		if !strings.HasSuffix(name, "-controller") {
			t.Errorf("name consistency check failed: controller %q must have \"-controller\" suffix", name)
		}
	}
}

func TestCloudControllerNamesDeclaration(t *testing.T) {
	declaredControllers := sets.New(
		names.CloudNodeController,
		names.ServiceLBController,
		names.NodeRouteController,
		names.CloudNodeLifecycleController,
	)

	for name := range DefaultInitFuncConstructors {
		if !declaredControllers.Has(name) {
			t.Errorf("name declaration check failed: controller name %q should be declared in  \"controller_names.go\" and added to this test", name)
		}
	}
}

func TestNativeHistogramsFeatureGateApplied(t *testing.T) {
	testCases := map[string]func(t *testing.T) *cobra.Command{
		"NewCloudControllerManagerCommand": func(t *testing.T) *cobra.Command {
			s, err := options.NewCloudControllerManagerOptions()
			if err != nil {
				t.Fatal(err)
			}
			return NewCloudControllerManagerCommand(s, nil, DefaultInitFuncConstructors, names.CCMControllerAliases(), cliflag.NamedFlagSets{}, wait.NeverStop)
		},
		"CommandBuilder": func(t *testing.T) *cobra.Command {
			cb := NewBuilder()
			cb.RegisterDefaultControllers()
			return cb.BuildCommand()
		},
	}

	originalReapplyHandling := logsapi.ReapplyHandling
	logsapi.ReapplyHandling = logsapi.ReapplyHandlingIgnoreUnchanged
	t.Cleanup(func() { logsapi.ReapplyHandling = originalReapplyHandling })

	for name, newCommand := range testCases {
		t.Run(name, func(t *testing.T) {
			// Registered before SetFeatureGateDuringTest so that it runs after the gate is restored.
			t.Cleanup(func() { features.ApplyFeatureGates(utilfeature.DefaultFeatureGate) })
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NativeHistograms, false)
			features.ApplyFeatureGates(utilfeature.DefaultFeatureGate)
			if histogramIsNative(t) {
				t.Fatal("histogram is native before the command has run")
			}

			// The kubeconfig does not exist, so the command fails in Config,
			// after it has applied the feature gates.
			cmd := newCommand(t)
			cmd.SetArgs([]string{
				"--kubeconfig=" + filepath.Join(t.TempDir(), "missing-kubeconfig"),
				"--feature-gates=NativeHistograms=true",
			})
			if err := cmd.Execute(); err == nil {
				t.Fatal("expected the command to fail")
			}

			if !histogramIsNative(t) {
				t.Error("histogram is not native although the NativeHistograms feature gate is enabled")
			}
		})
	}
}

func histogramIsNative(t *testing.T) bool {
	t.Helper()
	h := metrics.NewHistogram(&metrics.HistogramOpts{Name: "test_histogram", Help: "test"})
	registry := metrics.NewKubeRegistry()
	registry.MustRegister(h)
	h.Observe(0.5)
	mfs, err := registry.Gather()
	if err != nil {
		t.Fatal(err)
	}
	for _, mf := range mfs {
		if mf.GetName() == "test_histogram" {
			return mf.GetMetric()[0].GetHistogram().Schema != nil
		}
	}
	t.Fatal("histogram not gathered")
	return false
}
