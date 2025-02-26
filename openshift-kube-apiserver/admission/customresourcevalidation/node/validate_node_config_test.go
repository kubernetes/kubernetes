package node

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"

	configv1 "github.com/openshift/api/config/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
)

func TestValidateConfigNodeForExtremeLatencyProfile(t *testing.T) {
	testCases := []struct {
		fromProfile  configv1.WorkerLatencyProfileType
		toProfile    configv1.WorkerLatencyProfileType
		shouldReject bool
	}{
		// no rejections
		{fromProfile: "", toProfile: "", shouldReject: false},
		{fromProfile: "", toProfile: configv1.DefaultUpdateDefaultReaction, shouldReject: false},
		{fromProfile: "", toProfile: configv1.MediumUpdateAverageReaction, shouldReject: false},
		{fromProfile: configv1.DefaultUpdateDefaultReaction, toProfile: "", shouldReject: false},
		{fromProfile: configv1.DefaultUpdateDefaultReaction, toProfile: configv1.DefaultUpdateDefaultReaction, shouldReject: false},
		{fromProfile: configv1.DefaultUpdateDefaultReaction, toProfile: configv1.MediumUpdateAverageReaction, shouldReject: false},
		{fromProfile: configv1.MediumUpdateAverageReaction, toProfile: "", shouldReject: false},
		{fromProfile: configv1.MediumUpdateAverageReaction, toProfile: configv1.DefaultUpdateDefaultReaction, shouldReject: false},
		{fromProfile: configv1.MediumUpdateAverageReaction, toProfile: configv1.MediumUpdateAverageReaction, shouldReject: false},
		{fromProfile: configv1.MediumUpdateAverageReaction, toProfile: configv1.LowUpdateSlowReaction, shouldReject: false},
		{fromProfile: configv1.LowUpdateSlowReaction, toProfile: configv1.MediumUpdateAverageReaction, shouldReject: false},
		{fromProfile: configv1.LowUpdateSlowReaction, toProfile: configv1.LowUpdateSlowReaction, shouldReject: false},

		// rejections
		{fromProfile: "", toProfile: configv1.LowUpdateSlowReaction, shouldReject: true},
		{fromProfile: configv1.DefaultUpdateDefaultReaction, toProfile: configv1.LowUpdateSlowReaction, shouldReject: true},
		{fromProfile: configv1.LowUpdateSlowReaction, toProfile: "", shouldReject: true},
		{fromProfile: configv1.LowUpdateSlowReaction, toProfile: configv1.DefaultUpdateDefaultReaction, shouldReject: true},
	}

	for _, testCase := range testCases {
		shouldStr := "should not be"
		if testCase.shouldReject {
			shouldStr = "should be"
		}
		testCaseName := fmt.Sprintf("update from profile %s to %s %s rejected", testCase.fromProfile, testCase.toProfile, shouldStr)
		t.Run(testCaseName, func(t *testing.T) {
			// config node objects
			oldObject := configv1.Node{
				Spec: configv1.NodeSpec{
					WorkerLatencyProfile: testCase.fromProfile,
				},
			}
			newObject := configv1.Node{
				Spec: configv1.NodeSpec{
					WorkerLatencyProfile: testCase.toProfile,
				},
			}

			fieldErr := validateConfigNodeForExtremeLatencyProfile(&oldObject, &newObject)
			assert.Equal(t, testCase.shouldReject, fieldErr != nil, "latency profile from %q to %q %s rejected", testCase.fromProfile, testCase.toProfile, shouldStr)

			if testCase.shouldReject {
				assert.Equal(t, "spec.workerLatencyProfile", fieldErr.Field, "field name during for latency profile should be spec.workerLatencyProfile")
				assert.Contains(t, fieldErr.Detail, testCase.fromProfile, "error message should contain %q", testCase.fromProfile)
				assert.Contains(t, fieldErr.Detail, testCase.toProfile, "error message should contain %q", testCase.toProfile)
			}
		})
	}
}

func TestValidateConfigNodeForMinimumKubeletVersion(t *testing.T) {
	testCases := []struct {
		name         string
		version      string
		shouldReject bool
		nodes        []*v1.Node
		nodeListErr  error
		errType      field.ErrorType
		errMsg       string
	}{
		// no rejections
		{
			name:         "should not reject when minimum kubelet version is empty",
			version:      "",
			shouldReject: false,
		},
		{
			name:         "should reject when min kubelet version bogus",
			version:      "bogus",
			shouldReject: true,
			nodes: []*v1.Node{
				{
					Status: v1.NodeStatus{
						NodeInfo: v1.NodeSystemInfo{
							KubeletVersion: "1.30.0",
						},
					},
				},
			},
			errType: field.ErrorTypeInvalid,
			errMsg:  "failed to parse submitted version bogus No Major.Minor.Patch elements found",
		},
		{
			name:         "should reject when kubelet version is bogus",
			version:      "1.30.0",
			shouldReject: true,
			nodes: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "node",
					},
					Status: v1.NodeStatus{
						NodeInfo: v1.NodeSystemInfo{
							KubeletVersion: "bogus",
						},
					},
				},
			},
			errType: field.ErrorTypeInvalid,
			errMsg:  "failed to parse node version bogus: No Major.Minor.Patch elements found",
		},
		{
			name:         "should reject when kubelet version is too old",
			version:      "1.30.0",
			shouldReject: true,
			nodes: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "node",
					},
					Status: v1.NodeStatus{
						NodeInfo: v1.NodeSystemInfo{
							KubeletVersion: "1.29.0",
						},
					},
				},
			},
			errType: field.ErrorTypeForbidden,
			errMsg:  "kubelet version is 1.29.0, which is lower than minimumKubeletVersion of 1.30.0",
		},
		{
			name:         "should reject when one kubelet version is too old",
			version:      "1.30.0",
			shouldReject: true,
			nodes: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "node",
					},
					Status: v1.NodeStatus{
						NodeInfo: v1.NodeSystemInfo{
							KubeletVersion: "1.30.0",
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "node2",
					},
					Status: v1.NodeStatus{
						NodeInfo: v1.NodeSystemInfo{
							KubeletVersion: "1.29.0",
						},
					},
				},
			},
			errType: field.ErrorTypeForbidden,
			errMsg:  "kubelet version is 1.29.0, which is lower than minimumKubeletVersion of 1.30.0",
		},
		{
			name:         "should not reject when kubelet version is equal",
			version:      "1.30.0",
			shouldReject: false,
			nodes: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "node",
					},
					Status: v1.NodeStatus{
						NodeInfo: v1.NodeSystemInfo{
							KubeletVersion: "1.30.0",
						},
					},
				},
			},
		},
		{
			name:         "should reject when min version incomplete",
			version:      "1.30",
			shouldReject: true,
			nodes: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "node",
					},
					Status: v1.NodeStatus{
						NodeInfo: v1.NodeSystemInfo{
							KubeletVersion: "1.30.0",
						},
					},
				},
			},
			errType: field.ErrorTypeInvalid,
			errMsg:  "failed to parse submitted version 1.30 No Major.Minor.Patch elements found",
		},
		{
			name:         "should reject when kubelet version incomplete",
			version:      "1.30.0",
			shouldReject: true,
			nodes: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "node",
					},
					Status: v1.NodeStatus{
						NodeInfo: v1.NodeSystemInfo{
							KubeletVersion: "1.30",
						},
					},
				},
			},
			errType: field.ErrorTypeInvalid,
			errMsg:  "failed to parse node version 1.30: No Major.Minor.Patch elements found",
		},
		{
			name:         "should not reject when kubelet version is new enough",
			version:      "1.30.0",
			shouldReject: false,
			nodes: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "node",
					},
					Status: v1.NodeStatus{
						NodeInfo: v1.NodeSystemInfo{
							KubeletVersion: "1.31.0",
						},
					},
				},
			},
		},
	}
	for _, testCase := range testCases {
		shouldStr := "should not be"
		if testCase.shouldReject {
			shouldStr = "should be"
		}
		t.Run(testCase.name, func(t *testing.T) {
			obj := configv1.Node{
				Spec: configv1.NodeSpec{
					MinimumKubeletVersion: testCase.version,
				},
			}
			v := &configNodeV1{
				nodeListerFn: fakeNodeLister(testCase.nodes),
				waitForNodeInformerSyncedFn: func() bool {
					return true
				},
				minimumKubeletVersionEnabled: true,
			}

			fieldErr := v.validateMinimumKubeletVersion(&obj)
			assert.Equal(t, testCase.shouldReject, fieldErr != nil, "minimum kubelet version %q %s rejected", testCase.version, shouldStr)
			if testCase.shouldReject {
				assert.Equal(t, "spec.minimumKubeletVersion", fieldErr.Field, "field name during for mininumKubeletVersion should be spec.mininumKubeletVersion")
				assert.Equal(t, fieldErr.Type, testCase.errType, "error type should be %q", testCase.errType)
				assert.Contains(t, fieldErr.Detail, testCase.errMsg, "error message should contain %q", testCase.errMsg)
			}
		})
	}
}

func fakeNodeLister(nodes []*v1.Node) func() corev1listers.NodeLister {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	for _, node := range nodes {
		_ = indexer.Add(node)
	}
	return func() corev1listers.NodeLister {
		return corev1listers.NewNodeLister(indexer)
	}
}
