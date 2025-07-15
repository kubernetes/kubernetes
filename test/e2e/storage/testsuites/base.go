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

package testsuites

import (
	"context"
	"flag"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/component-base/metrics/testutil"
	csitrans "k8s.io/csi-translation-lib"
	"k8s.io/kubernetes/test/e2e/framework"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
)

var migratedPlugins *string

func init() {
	migratedPlugins = flag.String("storage.migratedPlugins", "", "comma separated list of in-tree plugin names of form 'kubernetes.io/{pluginName}' migrated to CSI")
}

type opCounts map[string]int64

// migrationOpCheck validates migrated metrics.
type migrationOpCheck struct {
	cs         clientset.Interface
	config     *rest.Config
	pluginName string
	skipCheck  bool

	// The old ops are not set if skipCheck is true.
	oldInTreeOps   opCounts
	oldMigratedOps opCounts
}

// BaseSuites is a list of storage test suites that work for in-tree and CSI drivers
var BaseSuites = []func() storageframework.TestSuite{
	InitCapacityTestSuite,
	InitVolumesTestSuite,
	InitVolumeIOTestSuite,
	InitVolumeModeTestSuite,
	InitSubPathTestSuite,
	InitProvisioningTestSuite,
	InitMultiVolumeTestSuite,
	InitVolumeExpandTestSuite,
	InitDisruptiveTestSuite,
	InitVolumeLimitsTestSuite,
	InitTopologyTestSuite,
	InitVolumeStressTestSuite,
	InitFsGroupChangePolicyTestSuite,
	InitVolumeGroupSnapshottableTestSuite,
	func() storageframework.TestSuite {
		return InitCustomEphemeralTestSuite(GenericEphemeralTestPatterns())
	},
}

// CSISuites is a list of storage test suites that work only for CSI drivers
var CSISuites = append(BaseSuites,
	func() storageframework.TestSuite {
		return InitCustomEphemeralTestSuite(CSIEphemeralTestPatterns())
	},
	InitSnapshottableTestSuite,
	InitVolumeGroupSnapshottableTestSuite,
	InitSnapshottableStressTestSuite,
	InitVolumePerformanceTestSuite,
	InitPvcDeletionPerformanceTestSuite,
	InitReadWriteOncePodTestSuite,
	InitVolumeModifyTestSuite,
	InitVolumeModifyStressTestSuite,
)

func getVolumeOpsFromMetricsForPlugin(ms testutil.Metrics, pluginName string) opCounts {
	totOps := opCounts{}

	for method, samples := range ms {
		switch method {
		case "storage_operation_status_count":
			for _, sample := range samples {
				plugin := string(sample.Metric["volume_plugin"])
				if pluginName != plugin {
					continue
				}
				opName := string(sample.Metric["operation_name"])
				if opName == "verify_controller_attached_volume" {
					// We ignore verify_controller_attached_volume because it does not call into
					// the plugin. It only watches Node API and updates Actual State of World cache
					continue
				}
				totOps[opName] = totOps[opName] + int64(sample.Value)
			}
		}
	}
	return totOps
}

func getVolumeOpCounts(ctx context.Context, c clientset.Interface, config *rest.Config, pluginName string) opCounts {
	if !framework.ProviderIs("gce", "aws") {
		return opCounts{}
	}

	nodeLimit := 25

	metricsGrabber, err := e2emetrics.NewMetricsGrabber(ctx, c, nil, config, true, false, true, false, false, false)

	if err != nil {
		framework.ExpectNoError(err, "Error creating metrics grabber: %v", err)
	}

	if !metricsGrabber.HasControlPlanePods() {
		framework.Logf("Warning: Environment does not support getting controller-manager metrics")
		return opCounts{}
	}

	controllerMetrics, err := metricsGrabber.GrabFromControllerManager(ctx)
	framework.ExpectNoError(err, "Error getting c-m metrics : %v", err)
	totOps := getVolumeOpsFromMetricsForPlugin(testutil.Metrics(controllerMetrics), pluginName)

	framework.Logf("Node name not specified for getVolumeOpCounts, falling back to listing nodes from API Server")
	nodes, err := c.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	framework.ExpectNoError(err, "Error listing nodes: %v", err)
	if len(nodes.Items) <= nodeLimit {
		// For large clusters with > nodeLimit nodes it is too time consuming to
		// gather metrics from all nodes. We just ignore the node metrics
		// for those clusters
		for _, node := range nodes.Items {
			nodeMetrics, err := metricsGrabber.GrabFromKubelet(ctx, node.GetName())
			framework.ExpectNoError(err, "Error getting Kubelet %v metrics: %v", node.GetName(), err)
			totOps = addOpCounts(totOps, getVolumeOpsFromMetricsForPlugin(testutil.Metrics(nodeMetrics), pluginName))
		}
	} else {
		framework.Logf("Skipping operation metrics gathering from nodes in getVolumeOpCounts, greater than %v nodes", nodeLimit)
	}

	return totOps
}

func addOpCounts(o1 opCounts, o2 opCounts) opCounts {
	totOps := opCounts{}
	seen := sets.NewString()
	for op, count := range o1 {
		seen.Insert(op)
		totOps[op] = totOps[op] + count + o2[op]
	}
	for op, count := range o2 {
		if !seen.Has(op) {
			totOps[op] = totOps[op] + count
		}
	}
	return totOps
}

func getMigrationVolumeOpCounts(ctx context.Context, cs clientset.Interface, config *rest.Config, pluginName string) (opCounts, opCounts) {
	if len(pluginName) > 0 {
		var migratedOps opCounts
		l := csitrans.New()
		csiName, err := l.GetCSINameFromInTreeName(pluginName)
		if err != nil {
			framework.Logf("Could not find CSI Name for in-tree plugin %v", pluginName)
			migratedOps = opCounts{}
		} else {
			csiName = "kubernetes.io/csi:" + csiName
			migratedOps = getVolumeOpCounts(ctx, cs, config, csiName)
		}
		return getVolumeOpCounts(ctx, cs, config, pluginName), migratedOps
	}
	// Not an in-tree driver
	framework.Logf("Test running for native CSI Driver, not checking metrics")
	return opCounts{}, opCounts{}
}

func newMigrationOpCheck(ctx context.Context, cs clientset.Interface, config *rest.Config, pluginName string) *migrationOpCheck {
	moc := migrationOpCheck{
		cs:         cs,
		config:     config,
		pluginName: pluginName,
	}
	if len(pluginName) == 0 {
		// This is a native CSI Driver and we don't check ops
		moc.skipCheck = true
		return &moc
	}

	if !sets.NewString(strings.Split(*migratedPlugins, ",")...).Has(pluginName) {
		// In-tree plugin is not migrated
		framework.Logf("In-tree plugin %v is not migrated, not validating any metrics", pluginName)

		// We don't check in-tree plugin metrics because some negative test
		// cases may not do any volume operations and therefore not emit any
		// metrics

		// We don't check counts for the Migrated version of the driver because
		// if tests are running in parallel a test could be using the CSI Driver
		// natively and increase the metrics count

		// TODO(dyzz): Add a dimension to OperationGenerator metrics for
		// "migrated"->true/false so that we can disambiguate migrated metrics
		// and native CSI Driver metrics. This way we can check the counts for
		// migrated version of the driver for stronger negative test case
		// guarantees (as well as more informative metrics).
		moc.skipCheck = true
		return &moc
	}

	// TODO: temporarily skip metrics check due to issue #[102893](https://github.com/kubernetes/kubernetes/issues/102893)
	// Will remove it once the issue is fixed
	if framework.NodeOSDistroIs("windows") {
		moc.skipCheck = true
		return &moc
	}

	moc.oldInTreeOps, moc.oldMigratedOps = getMigrationVolumeOpCounts(ctx, cs, config, pluginName)
	return &moc
}

func (moc *migrationOpCheck) validateMigrationVolumeOpCounts(ctx context.Context) {
	if moc.skipCheck {
		return
	}

	newInTreeOps, _ := getMigrationVolumeOpCounts(ctx, moc.cs, moc.config, moc.pluginName)

	for op, count := range newInTreeOps {
		if count != moc.oldInTreeOps[op] {
			framework.Failf("In-tree plugin %v migrated to CSI Driver, however found %v %v metrics for in-tree plugin", moc.pluginName, count-moc.oldInTreeOps[op], op)
		}
	}
	// We don't check for migrated metrics because some negative test cases
	// may not do any volume operations and therefore not emit any metrics
}

// Skip skipVolTypes patterns if the driver supports dynamic provisioning
func skipVolTypePatterns(pattern storageframework.TestPattern, driver storageframework.TestDriver, skipVolTypes map[storageframework.TestVolType]bool) {
	_, supportsProvisioning := driver.(storageframework.DynamicPVTestDriver)
	if supportsProvisioning && skipVolTypes[pattern.VolType] {
		e2eskipper.Skipf("Driver supports dynamic provisioning, skipping %s pattern", pattern.VolType)
	}
}
