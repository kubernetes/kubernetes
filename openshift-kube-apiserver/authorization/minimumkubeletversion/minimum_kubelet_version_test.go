package minimumkubeletversion

import (
	"context"
	"strings"
	"testing"

	"github.com/blang/semver/v4"
	authorizationv1 "k8s.io/api/authorization/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/user"
	kauthorizer "k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/auth/nodeidentifier"
	"k8s.io/kubernetes/pkg/controller"
)

func TestAuthorize(t *testing.T) {
	nodeUser := &user.DefaultInfo{Name: "system:node:node0", Groups: []string{"system:nodes"}}

	testCases := []struct {
		name            string
		minVersion      string
		attributes      kauthorizer.AttributesRecord
		expectedAllowed kauthorizer.Decision
		expectedErr     string
		expectedMsg     string
		node            *v1.Node
	}{
		{
			name:            "no version",
			minVersion:      "",
			expectedAllowed: kauthorizer.DecisionNoOpinion,
			expectedErr:     "",
			node:            &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "name"}},
		},
		{
			name:       "user not a node",
			minVersion: "1.30.0",
			attributes: kauthorizer.AttributesRecord{
				ResourceRequest: true,
				Namespace:       "ns",
				User:            &user.DefaultInfo{Name: "name"},
			},
			expectedAllowed: kauthorizer.DecisionNoOpinion,
			node:            &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node0"}},
		},
		{
			name:       "skips if subjectaccessreviews",
			minVersion: "1.30.0",
			attributes: kauthorizer.AttributesRecord{
				ResourceRequest: true,
				Namespace:       "ns",
				User:            nodeUser,
				Resource:        "subjectaccessreviews",
				APIGroup:        authorizationv1.GroupName,
			},
			expectedAllowed: kauthorizer.DecisionNoOpinion,
			node:            &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node0"}},
		},
		{
			name:       "skips if get node",
			minVersion: "1.30.0",
			attributes: kauthorizer.AttributesRecord{
				ResourceRequest: true,
				Namespace:       "ns",
				User:            nodeUser,
				Resource:        "nodes",
				Verb:            "get",
			},
			expectedAllowed: kauthorizer.DecisionNoOpinion,
			node:            &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node0"}},
		},
		{
			name:       "skips if update nodes",
			minVersion: "1.30.0",
			attributes: kauthorizer.AttributesRecord{
				ResourceRequest: true,
				Namespace:       "ns",
				User:            nodeUser,
				Resource:        "nodes",
				Verb:            "update",
			},
			expectedAllowed: kauthorizer.DecisionNoOpinion,
			node:            &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node0"}},
		},
		{
			name:       "fail if update node not found",
			minVersion: "1.30.0",
			attributes: kauthorizer.AttributesRecord{
				ResourceRequest: true,
				Namespace:       "ns",
				User:            nodeUser,
			},
			expectedAllowed: kauthorizer.DecisionDeny,
			expectedErr:     `node "node0" not found`,
			node:            &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node1"}},
		},
		{
			name:       "skip if bogus kubelet version",
			minVersion: "1.30.0",
			attributes: kauthorizer.AttributesRecord{
				ResourceRequest: true,
				Namespace:       "ns",
				User:            nodeUser,
			},
			expectedAllowed: kauthorizer.DecisionDeny,
			expectedErr:     `failed to parse node version bogus: No Major.Minor.Patch elements found`,
			node: &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node0"},
				Status: v1.NodeStatus{
					NodeInfo: v1.NodeSystemInfo{
						KubeletVersion: "bogus",
					},
				}},
		},
		{
			name:       "deny if too low version",
			minVersion: "1.30.0",
			attributes: kauthorizer.AttributesRecord{
				ResourceRequest: true,
				Namespace:       "ns",
				User:            nodeUser,
			},
			expectedAllowed: kauthorizer.DecisionDeny,
			expectedMsg:     `kubelet version is outdated: kubelet version is 1.29.8, which is lower than minimumKubeletVersion of 1.30.0`,
			node: &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node0"},
				Status: v1.NodeStatus{
					NodeInfo: v1.NodeSystemInfo{
						KubeletVersion: "v1.29.8-20+15d27f9ba1c119",
					},
				}},
		},
		{
			name:       "accept if high enough version",
			minVersion: "1.30.0",
			attributes: kauthorizer.AttributesRecord{
				ResourceRequest: true,
				Namespace:       "ns",
				User:            nodeUser,
			},
			expectedAllowed: kauthorizer.DecisionNoOpinion,
			node: &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node0"},
				Status: v1.NodeStatus{
					NodeInfo: v1.NodeSystemInfo{
						KubeletVersion: "1.30.0",
					},
				}},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			fakeInformerFactory := informers.NewSharedInformerFactory(&fake.Clientset{}, controller.NoResyncPeriodFunc())
			fakeNodeInformer := fakeInformerFactory.Core().V1().Nodes()
			fakeNodeInformer.Informer().GetStore().Add(tc.node)
			var minVersion *semver.Version
			if tc.minVersion != "" {
				v := semver.MustParse(tc.minVersion)
				minVersion = &v
			}

			authorizer := &minimumKubeletVersionAuth{
				nodeIdentifier: nodeidentifier.NewDefaultNodeIdentifier(),
				nodeLister:     fakeNodeInformer.Lister(),
				minVersion:     minVersion,
				hasNodeInformerSyncedFn: func() bool {
					return true
				},
			}

			actualAllowed, actualMsg, actualErr := authorizer.Authorize(context.TODO(), tc.attributes)
			switch {
			case len(tc.expectedErr) == 0 && actualErr == nil:
			case len(tc.expectedErr) == 0 && actualErr != nil:
				t.Errorf("%s: unexpected error: %v", tc.name, actualErr)
			case len(tc.expectedErr) != 0 && actualErr == nil:
				t.Errorf("%s: missing error: %v", tc.name, tc.expectedErr)
			case len(tc.expectedErr) != 0 && actualErr != nil:
				if !strings.Contains(actualErr.Error(), tc.expectedErr) {
					t.Errorf("expected %v, got %v", tc.expectedErr, actualErr)
				}
			}
			if tc.expectedMsg != actualMsg {
				t.Errorf("expected %v, got %v", tc.expectedMsg, actualMsg)
			}
			if tc.expectedAllowed != actualAllowed {
				t.Errorf("expected %v, got %v", tc.expectedAllowed, actualAllowed)
			}
		})
	}
}
