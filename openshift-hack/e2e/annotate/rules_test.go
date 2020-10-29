package annotate

import (
	"testing"

	"github.com/onsi/ginkgo/v2/types"
)

type testNode struct {
	text string
}

func (n *testNode) CodeLocations() []types.CodeLocation {
	return []types.CodeLocation{{FileName: "k8s.io/kubernetes"}}
}

func (n *testNode) Text() string {
	return n.text
}

func (n *testNode) AppendText(text string) {
	n.text += text
}

func TestStockRules(t *testing.T) {
	tests := []struct {
		name string

		testName string

		expectedLabel string
		expectedText  string
	}{
		{
			name:          "simple serial match",
			testName:      "[Serial] test",
			expectedLabel: " [Suite:openshift/conformance/serial]",
			expectedText:  "[Serial] test [Suite:openshift/conformance/serial]",
		},
		{
			name:          "don't tag skipped",
			testName:      `[Serial] example test [Skipped:gce]`,
			expectedLabel: ` [Suite:openshift/conformance/serial]`,
			expectedText:  `[Serial] example test [Skipped:gce] [Suite:openshift/conformance/serial]`, // notice that this isn't categorized into any of our buckets
		},
		{
			name:          "not skipped",
			testName:      `[sig-network] Networking Granular Checks: Pods should function for intra-pod communication: http [LinuxOnly] [NodeConformance] [Conformance]`,
			expectedLabel: ` [Suite:openshift/conformance/parallel/minimal]`,
			expectedText:  `[sig-network] Networking Granular Checks: Pods should function for intra-pod communication: http [LinuxOnly] [NodeConformance] [Conformance] [Suite:openshift/conformance/parallel/minimal]`,
		},
		{
			name:          "should skip localssd on gce",
			testName:      `[sig-storage] In-tree Volumes [Driver: local][LocalVolumeType: gce-localssd-scsi-fs] [Serial] [Testpattern: Dynamic PV (default fs)] subPath should be able to unmount after the subpath directory is deleted`,
			expectedLabel: ` [Skipped:gce] [Suite:openshift/conformance/serial]`,
			expectedText:  `[sig-storage] In-tree Volumes [Driver: local][LocalVolumeType: gce-localssd-scsi-fs] [Serial] [Testpattern: Dynamic PV (default fs)] subPath should be able to unmount after the subpath directory is deleted [Skipped:gce] [Suite:openshift/conformance/serial]`, // notice that this isn't categorized into any of our buckets
		},
		{
			name:          "should skip NetworkPolicy tests on multitenant",
			testName:      `should do something with NetworkPolicy`,
			expectedLabel: ` [Suite:openshift/conformance/parallel]`,
			expectedText:  `should do something with NetworkPolicy [Suite:openshift/conformance/parallel]`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			testRenamer := newGenerator(TestMaps)
			testNode := &testNode{
				text: test.testName,
			}

			testRenamer.generateRename(test.testName, testNode)
			changed := testRenamer.output[test.testName]

			if e, a := test.expectedLabel, changed; e != a {
				t.Error(a)
			}
			testRenamer = newRenamerFromGenerated(map[string]string{test.testName: test.expectedLabel})
			testRenamer.updateNodeText(test.testName, testNode)

			if e, a := test.expectedText, testNode.Text(); e != a {
				t.Logf(e)
				t.Error(a)
			}
		})
	}
}
