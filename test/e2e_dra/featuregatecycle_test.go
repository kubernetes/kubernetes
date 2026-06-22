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

package e2edra

import (
	"strings"
	"testing"
	"time"

	"github.com/onsi/gomega"

	restclient "k8s.io/client-go/rest"
	"k8s.io/dynamic-resource-allocation/resourceslice"
	drautils "k8s.io/kubernetes/test/e2e/dra/utils"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2etestfiles "k8s.io/kubernetes/test/e2e/framework/testfiles"
	"k8s.io/kubernetes/test/utils/client-go/ktesting"
	"k8s.io/kubernetes/test/utils/localupcluster"
)

// gateOnFunc runs while the feature gate is ON and returns the next phase.
type gateOnFunc func(tCtx ktesting.TContext, b *drautils.Builder) gateOffFunc

// gateOffFunc runs while the feature gate is OFF and returns the next phase.
type gateOffFunc func(tCtx ktesting.TContext) gateOnAgainFunc

// gateOnAgainFunc runs after the feature gate is re-enabled.
type gateOnAgainFunc func(tCtx ktesting.TContext)

type getResources func(nodes *drautils.Nodes) map[string]resourceslice.DriverResources

func TestFeatureGateCycle(t *testing.T) { testFeatureGateCycle(ktesting.Init(t)) }

func testFeatureGateCycle(tCtx ktesting.TContext) {
	e2etestfiles.AddFileSource(e2etestfiles.RootFileSource{Root: repoRoot})

	gomega.RegisterFailHandler(func(message string, callerSkip ...int) {
		tCtx.Helper()
		tCtx.Fatal(message)
	})

	envName, dir := currentBinDir()
	if dir == "" {
		tCtx.Fatalf("%s must be set", envName)
	}

	subTests := map[string]struct {
		test                gateOnFunc
		driverResourcesFunc getResources
		// gates lists the feature gates to start ON in phase 0, toggle OFF in phase 1, and re-enable in phase 2.
		gates []string
		// emulatedVersion, if non-empty, is passed as EMULATED_VERSION to local-up-cluster.sh.
		// Set this to the last version where the gates were still toggleable so the test
		// keeps working once those gates become locked in a future release.
		emulatedVersion string
	}{
		"extended-resource": {
			test:                extendedResourceGateCycle,
			driverResourcesFunc: extendedResourcesDriverResources,
			gates:               []string{"DRAExtendedResource"},
			// Emulate 1.36 so that gates which are locked in the current binary
			// remain toggleable. DRAExtendedResource is toggleable in 1.36.
			emulatedVersion: "1.36",
		},
	}

	for name, def := range subTests {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			// ---- Phase 0: gate(s) ON ----
			cluster := localupcluster.New()
			tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
				tCtx.Step("cleanup", cluster.Stop)
			})
			localUpClusterEnv := map[string]string{
				"RUNTIME_CONFIG": localUpClusterRuntimeConfig,
			}
			if def.emulatedVersion != "" {
				localUpClusterEnv["EMULATED_VERSION"] = def.emulatedVersion
			}
			phase := "phase-0-gate-on"
			cluster.Start(tCtx, phase, dir, localUpClusterEnv, turnFgOn(def.gates))

			restConfig := cluster.LoadConfig(tCtx)
			restConfig.UserAgent = restclient.DefaultKubernetesUserAgent() + " -- dra"
			tCtx = tCtx.WithRESTConfig(restConfig).WithNamespace("default")

			var nodes *drautils.Nodes
			tCtx.Step("wait for node", func(tCtx ktesting.TContext) {
				tCtx.ExpectNoError(e2enode.WaitForAllNodesSchedulable(tCtx, tCtx.Client(), 5*time.Minute))
				nodes = drautils.NewNodesNow(tCtx, 1, 1)
			})

			// Create builder
			driver := drautils.NewDriverInstance(tCtx)
			driver.SetNameSuffix(tCtx, name)
			driver.IsLocal = true
			driver.Run(tCtx, "/var/lib/kubelet", nodes, def.driverResourcesFunc(nodes))
			builder := drautils.NewBuilderNow(tCtx, driver)
			builder.SkipCleanup = true

			var gateOffFn gateOffFunc
			tCtx.Run(phase, func(tCtx ktesting.TContext) {
				gateOffFn = def.test(tCtx, builder)
			})

			// ---- Phase 1: gate(s) OFF ----
			phase = "phase-1-gate-off"
			cluster.ToggleFeatureGates(tCtx, phase, turnFgOff(def.gates))
			waitForSlices(tCtx, name, builder, def.driverResourcesFunc(nodes))

			var gateOnAgainFn gateOnAgainFunc
			tCtx.Run(phase, func(tCtx ktesting.TContext) {
				gateOnAgainFn = gateOffFn(tCtx)
			})

			// ---- Phase 2: gate(s) ON again ----
			phase = "phase-2-gate-on-again"
			cluster.ToggleFeatureGates(tCtx, phase, turnFgOn(def.gates))
			waitForSlices(tCtx, name, builder, def.driverResourcesFunc(nodes))

			tCtx.Run(phase, func(tCtx ktesting.TContext) {
				gateOnAgainFn(tCtx)
			})
		})
	}
}

// turnFgOn returns a string to turn the given feature gates on, e.g. "FeatureA=true,FeatureB=true".
func turnFgOn(featureGates []string) string {
	return turnFg(featureGates, "=true")
}

// turnFgOff returns a string to turn the given feature gates off, e.g. "FeatureA=false,FeatureB=false".
func turnFgOff(featureGates []string) string {
	return turnFg(featureGates, "=false")
}

// turnFg is a common part of turnFgOn and turnFgOff, returns a string to turn the given
// feature gates on or off based on the suffix, e.g. "FeatureA=true,FeatureB=true" or "FeatureA=false,FeatureB=false".
func turnFg(featureGates []string, suffix string) string {
	if len(featureGates) == 0 {
		return ""
	}
	return strings.Join(featureGates, suffix+",") + suffix
}
