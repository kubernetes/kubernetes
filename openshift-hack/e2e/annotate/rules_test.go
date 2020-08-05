package annotate

import (
	"testing"

	"github.com/onsi/ginkgo/types"
)

type testNode struct {
	text string
}

func (n *testNode) Type() types.SpecComponentType {
	return 0
}
func (n *testNode) CodeLocation() types.CodeLocation {
	return types.CodeLocation{}
}
func (n *testNode) Text() string {
	return n.text
}
func (n *testNode) SetText(text string) {
	n.text = text
}
func (n *testNode) Flag() types.FlagType {
	return 0
}
func (n *testNode) SetFlag(flag types.FlagType) {
}

func TestStockRules(t *testing.T) {
	tests := []struct {
		name string

		testName   string
		parentName string

		expectedText string
	}{
		{
			name:         "simple serial match",
			parentName:   "",
			testName:     "[Serial] test",
			expectedText: "[Serial] test [Suite:openshift/conformance/serial]",
		},
		{
			name:         "don't tag skipped",
			parentName:   "",
			testName:     `[Serial] example test [Skipped:gce]`,
			expectedText: `[Serial] example test [Skipped:gce] [Suite:openshift/conformance/serial]`, // notice that this isn't categorized into any of our buckets
		},
		{
			name:         "not skipped",
			parentName:   "",
			testName:     `[sig-network] Networking Granular Checks: Pods should function for intra-pod communication: http [LinuxOnly] [NodeConformance] [Conformance]`,
			expectedText: `[sig-network] Networking Granular Checks: Pods should function for intra-pod communication: http [LinuxOnly] [NodeConformance] [Conformance] [Suite:openshift/conformance/parallel/minimal]`,
		},
		{
			name:         "should skip localssd on gce",
			parentName:   "",
			testName:     `[sig-storage] In-tree Volumes [Driver: local][LocalVolumeType: gce-localssd-scsi-fs] [Serial] [Testpattern: Dynamic PV (default fs)] subPath should be able to unmount after the subpath directory is deleted`,
			expectedText: `[sig-storage] In-tree Volumes [Driver: local][LocalVolumeType: gce-localssd-scsi-fs] [Serial] [Testpattern: Dynamic PV (default fs)] subPath should be able to unmount after the subpath directory is deleted [Skipped:gce] [Suite:openshift/conformance/serial]`, // notice that this isn't categorized into any of our buckets
		},
		{
			name:         "should skip NetworkPolicy tests on multitenant",
			parentName:   "[Feature:NetworkPolicy]",
			testName:     `should do something with NetworkPolicy`,
			expectedText: `should do something with NetworkPolicy [Skipped:Network/OpenShiftSDN/Multitenant] [Suite:openshift/conformance/parallel]`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			testRenamer := newGenerator()
			testNode := &testNode{
				text: test.testName,
			}

			testRenamer.generateRename(test.testName, test.parentName, testNode)
			changed := testRenamer.output[combineNames(test.parentName, test.testName)]

			if e, a := test.expectedText, changed; e != a {
				t.Error(a)
			}
			testRenamer = newRenamerFromGenerated(map[string]string{combineNames(test.parentName, test.testName): test.expectedText})
			testRenamer.updateNodeText(test.testName, test.parentName, testNode)

			if e, a := test.expectedText, testNode.Text(); e != a {
				t.Error(a)
			}

		})
	}
}
