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

package testsuites

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/types"

	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	storageutils "k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

type capacityTestSuite struct {
	tsInfo storageframework.TestSuiteInfo
}

// InitCustomCapacityTestSuite returns capacityTestSuite that implements TestSuite interface
// using custom test patterns
func InitCustomCapacityTestSuite(patterns []storageframework.TestPattern) storageframework.TestSuite {
	return &capacityTestSuite{
		tsInfo: storageframework.TestSuiteInfo{
			Name:         "capacity",
			TestPatterns: patterns,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Mi",
			},
		},
	}
}

// InitCapacityTestSuite returns capacityTestSuite that implements TestSuite interface\
// using test suite default patterns
func InitCapacityTestSuite() storageframework.TestSuite {
	patterns := []storageframework.TestPattern{
		storageframework.DefaultFsDynamicPV,
	}
	return InitCustomCapacityTestSuite(patterns)
}

func (p *capacityTestSuite) GetTestSuiteInfo() storageframework.TestSuiteInfo {
	return p.tsInfo
}

func (p *capacityTestSuite) SkipUnsupportedTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	// Check preconditions.
	if pattern.VolType != storageframework.DynamicPV {
		e2eskipper.Skipf("Suite %q does not support %v", p.tsInfo.Name, pattern.VolType)
	}
	dInfo := driver.GetDriverInfo()
	if !dInfo.Capabilities[storageframework.CapCapacity] {
		e2eskipper.Skipf("Driver %s doesn't publish storage capacity -- skipping", dInfo.Name)
	}
}

func (p *capacityTestSuite) DefineTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	var (
		dInfo         = driver.GetDriverInfo()
		dDriver       storageframework.DynamicPVTestDriver
		driverCleanup func()
		sc            *storagev1.StorageClass
	)

	// Beware that it also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewFrameworkWithCustomTimeouts("capacity", storageframework.GetDriverTimeouts(driver))
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	init := func() {
		dDriver, _ = driver.(storageframework.DynamicPVTestDriver)
		// Now do the more expensive test initialization.
		config, cleanup := driver.PrepareTest(f)
		driverCleanup = cleanup
		sc = dDriver.GetDynamicProvisionStorageClass(config, pattern.FsType)
		if sc == nil {
			e2eskipper.Skipf("Driver %q does not define Dynamic Provision StorageClass - skipping", dInfo.Name)
		}
	}

	cleanup := func() {
		err := storageutils.TryFunc(driverCleanup)
		driverCleanup = nil
		framework.ExpectNoError(err, "while cleaning up driver")
	}

	ginkgo.It("provides storage capacity information", func() {
		init()
		defer cleanup()

		timeout := time.Minute
		pollInterval := time.Second
		matchSC := HaveCapacitiesForClass(sc.Name)
		listAll := gomega.Eventually(func() (*storagev1.CSIStorageCapacityList, error) {
			return f.ClientSet.StorageV1().CSIStorageCapacities("").List(context.Background(), metav1.ListOptions{})
		}, timeout, pollInterval)

		// If we have further information about what storage
		// capacity information to expect from the driver,
		// then we can make the check more specific. The baseline
		// is that it provides some arbitrary capacity for the
		// storage class.
		matcher := matchSC
		if len(dInfo.TopologyKeys) == 1 {
			// We can construct topology segments by
			// collecting all values for this one key and
			// then expect one CSIStorageCapacity object
			// per value for the storage class.
			//
			// Local storage on a node will be covered by
			// this checking. A more complex approach for
			// drivers with multiple keys might be
			// possible, too, but is not currently
			// implemented.
			matcher = HaveCapacitiesForClassAndNodes(f.ClientSet, sc.Provisioner, sc.Name, dInfo.TopologyKeys[0])
		}

		// Create storage class and wait for capacity information.
		_, clearProvisionedStorageClass := SetupStorageClass(f.ClientSet, sc)
		defer clearProvisionedStorageClass()
		listAll.Should(MatchCapacities(matcher), "after creating storage class")

		// Delete storage class again and wait for removal of storage capacity information.
		clearProvisionedStorageClass()
		listAll.ShouldNot(MatchCapacities(matchSC), "after deleting storage class")
	})
}

func formatCapacities(capacities []storagev1.CSIStorageCapacity) []string {
	lines := []string{}
	for _, capacity := range capacities {
		lines = append(lines, fmt.Sprintf("   %+v", capacity))
	}
	return lines
}

// MatchCapacities runs some kind of check against *storagev1.CSIStorageCapacityList.
// In case of failure, all actual objects are appended to the failure message.
func MatchCapacities(match types.GomegaMatcher) types.GomegaMatcher {
	return matchCSIStorageCapacities{match: match}
}

type matchCSIStorageCapacities struct {
	match types.GomegaMatcher
}

var _ types.GomegaMatcher = matchCSIStorageCapacities{}

func (m matchCSIStorageCapacities) Match(actual interface{}) (success bool, err error) {
	return m.match.Match(actual)
}

func (m matchCSIStorageCapacities) FailureMessage(actual interface{}) (message string) {
	return m.match.FailureMessage(actual) + m.dump(actual)
}

func (m matchCSIStorageCapacities) NegatedFailureMessage(actual interface{}) (message string) {
	return m.match.NegatedFailureMessage(actual) + m.dump(actual)
}

func (m matchCSIStorageCapacities) dump(actual interface{}) string {
	capacities, ok := actual.(*storagev1.CSIStorageCapacityList)
	if !ok || capacities == nil {
		return ""
	}
	lines := []string{"\n\nall CSIStorageCapacity objects:"}
	for _, capacity := range capacities.Items {
		lines = append(lines, fmt.Sprintf("%+v", capacity))
	}
	return strings.Join(lines, "\n")
}

// CapacityMatcher can be used to compose different matchers where one
// adds additional checks for CSIStorageCapacity objects already checked
// by another.
type CapacityMatcher interface {
	types.GomegaMatcher
	// MatchedCapacities returns all CSICapacityObjects which were
	// found during the preceding Match call.
	MatchedCapacities() []storagev1.CSIStorageCapacity
}

// HaveCapacitiesForClass filters all storage capacity objects in a *storagev1.CSIStorageCapacityList
// by storage class. Success is when when there is at least one.
func HaveCapacitiesForClass(scName string) CapacityMatcher {
	return &haveCSIStorageCapacities{scName: scName}
}

type haveCSIStorageCapacities struct {
	scName             string
	matchingCapacities []storagev1.CSIStorageCapacity
}

var _ CapacityMatcher = &haveCSIStorageCapacities{}

func (h *haveCSIStorageCapacities) Match(actual interface{}) (success bool, err error) {
	capacities, ok := actual.(*storagev1.CSIStorageCapacityList)
	if !ok {
		return false, fmt.Errorf("expected *storagev1.CSIStorageCapacityList, got: %T", actual)
	}
	h.matchingCapacities = nil
	for _, capacity := range capacities.Items {
		if capacity.StorageClassName == h.scName {
			h.matchingCapacities = append(h.matchingCapacities, capacity)
		}
	}
	return len(h.matchingCapacities) > 0, nil
}

func (h *haveCSIStorageCapacities) MatchedCapacities() []storagev1.CSIStorageCapacity {
	return h.matchingCapacities
}

func (h *haveCSIStorageCapacities) FailureMessage(actual interface{}) (message string) {
	return fmt.Sprintf("no CSIStorageCapacity objects for storage class %q", h.scName)
}

func (h *haveCSIStorageCapacities) NegatedFailureMessage(actual interface{}) (message string) {
	return fmt.Sprintf("CSIStorageCapacity objects for storage class %q:\n%s",
		h.scName,
		strings.Join(formatCapacities(h.matchingCapacities), "\n"),
	)
}

// HaveCapacitiesForClassAndNodes matches objects by storage class name. It finds
// all nodes on which the driver runs and expects one object per node.
func HaveCapacitiesForClassAndNodes(client kubernetes.Interface, driverName, scName, topologyKey string) CapacityMatcher {
	return &haveLocalStorageCapacities{
		client:      client,
		driverName:  driverName,
		match:       HaveCapacitiesForClass(scName),
		topologyKey: topologyKey,
	}
}

type haveLocalStorageCapacities struct {
	client      kubernetes.Interface
	driverName  string
	match       CapacityMatcher
	topologyKey string

	matchSuccess          bool
	expectedCapacities    []storagev1.CSIStorageCapacity
	unexpectedCapacities  []storagev1.CSIStorageCapacity
	missingTopologyValues []string
}

var _ CapacityMatcher = &haveLocalStorageCapacities{}

func (h *haveLocalStorageCapacities) Match(actual interface{}) (success bool, err error) {
	h.expectedCapacities = nil
	h.unexpectedCapacities = nil
	h.missingTopologyValues = nil

	// First check with underlying matcher.
	success, err = h.match.Match(actual)
	h.matchSuccess = success
	if !success || err != nil {
		return
	}

	// Find all nodes on which the driver runs.
	csiNodes, err := h.client.StorageV1().CSINodes().List(context.Background(), metav1.ListOptions{})
	if err != nil {
		return false, err
	}
	topologyValues := map[string]bool{}
	for _, csiNode := range csiNodes.Items {
		for _, driver := range csiNode.Spec.Drivers {
			if driver.Name != h.driverName {
				continue
			}
			node, err := h.client.CoreV1().Nodes().Get(context.Background(), csiNode.Name, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			value, ok := node.Labels[h.topologyKey]
			if !ok || value == "" {
				return false, fmt.Errorf("driver %q should run on node %q, but its topology label %q was not set",
					h.driverName,
					node.Name,
					h.topologyKey)
			}
			topologyValues[value] = true
			break
		}
	}
	if len(topologyValues) == 0 {
		return false, fmt.Errorf("driver %q not running on any node", h.driverName)
	}

	// Now check that for each topology value there is exactly one CSIStorageCapacity object.
	remainingTopologyValues := map[string]bool{}
	for value := range topologyValues {
		remainingTopologyValues[value] = true
	}
	capacities := h.match.MatchedCapacities()
	for _, capacity := range capacities {
		if capacity.NodeTopology == nil ||
			len(capacity.NodeTopology.MatchExpressions) > 0 ||
			len(capacity.NodeTopology.MatchLabels) != 1 ||
			!remainingTopologyValues[capacity.NodeTopology.MatchLabels[h.topologyKey]] {
			h.unexpectedCapacities = append(h.unexpectedCapacities, capacity)
			continue
		}
		remainingTopologyValues[capacity.NodeTopology.MatchLabels[h.topologyKey]] = false
		h.expectedCapacities = append(h.expectedCapacities, capacity)
	}

	// Success is when there were no unexpected capacities and enough expected ones.
	for value, remaining := range remainingTopologyValues {
		if remaining {
			h.missingTopologyValues = append(h.missingTopologyValues, value)
		}
	}
	return len(h.unexpectedCapacities) == 0 && len(h.missingTopologyValues) == 0, nil
}

func (h *haveLocalStorageCapacities) MatchedCapacities() []storagev1.CSIStorageCapacity {
	return h.match.MatchedCapacities()
}

func (h *haveLocalStorageCapacities) FailureMessage(actual interface{}) (message string) {
	if !h.matchSuccess {
		return h.match.FailureMessage(actual)
	}
	var lines []string
	if len(h.unexpectedCapacities) != 0 {
		lines = append(lines, "unexpected CSIStorageCapacity objects:")
		lines = append(lines, formatCapacities(h.unexpectedCapacities)...)
	}
	if len(h.missingTopologyValues) != 0 {
		lines = append(lines, fmt.Sprintf("no CSIStorageCapacity objects with topology key %q and values %v",
			h.topologyKey, h.missingTopologyValues,
		))
	}
	return strings.Join(lines, "\n")
}

func (h *haveLocalStorageCapacities) NegatedFailureMessage(actual interface{}) (message string) {
	if h.matchSuccess {
		return h.match.NegatedFailureMessage(actual)
	}
	// It's not entirely clear whether negating this check is useful. Just dump all info that we have.
	var lines []string
	if len(h.expectedCapacities) != 0 {
		lines = append(lines, "expected CSIStorageCapacity objects:")
		lines = append(lines, formatCapacities(h.expectedCapacities)...)
	}
	if len(h.unexpectedCapacities) != 0 {
		lines = append(lines, "unexpected CSIStorageCapacity objects:")
		lines = append(lines, formatCapacities(h.unexpectedCapacities)...)
	}
	if len(h.missingTopologyValues) != 0 {
		lines = append(lines, fmt.Sprintf("no CSIStorageCapacity objects with topology key %q and values %v",
			h.topologyKey, h.missingTopologyValues,
		))
	}
	return strings.Join(lines, "\n")
}
