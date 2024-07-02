package clusterrole

import (
	"testing"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/rbac"
)

func TestUpdateRulesForAggregation(t *testing.T) {
	s := strategy{}

	testCases := []struct {
		name            string
		oldClusterRole  *rbac.ClusterRole
		newClusterRole  *rbac.ClusterRole
		expectSameAsOld bool
		expectSameAsNew bool
	}{
		{
			name: "no change",
			oldClusterRole: &rbac.ClusterRole{
				AggregationRule: &rbac.AggregationRule{
					ClusterRoleSelectors: []metav1.LabelSelector{
						{
							MatchLabels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
			},
			newClusterRole: &rbac.ClusterRole{
				AggregationRule: &rbac.AggregationRule{
					ClusterRoleSelectors: []metav1.LabelSelector{
						{
							MatchLabels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
			},
			expectSameAsOld: true,
			expectSameAsNew: true,
		},
		{
			name: "no change with rules",
			oldClusterRole: &rbac.ClusterRole{
				AggregationRule: &rbac.AggregationRule{
					ClusterRoleSelectors: []metav1.LabelSelector{
						{
							MatchLabels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
				Rules: []rbac.PolicyRule{
					{
						Verbs:     []string{"get"},
						APIGroups: []string{"*"},
						Resources: []string{"pods"},
					},
				},
			},
			newClusterRole: &rbac.ClusterRole{
				AggregationRule: &rbac.AggregationRule{
					ClusterRoleSelectors: []metav1.LabelSelector{
						{
							MatchLabels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
				Rules: []rbac.PolicyRule{
					{
						Verbs:     []string{"get"},
						APIGroups: []string{"*"},
						Resources: []string{"pods"},
					},
				},
			},
			expectSameAsOld: true,
			expectSameAsNew: true,
		},
		{
			name: "no change with rules and aggregation rule nil",
			oldClusterRole: &rbac.ClusterRole{
				Rules: []rbac.PolicyRule{
					{
						Verbs:     []string{"get"},
						APIGroups: []string{"*"},
						Resources: []string{"pods"},
					},
				},
			},
			newClusterRole: &rbac.ClusterRole{
				Rules: []rbac.PolicyRule{
					{
						Verbs:     []string{"get"},
						APIGroups: []string{"*"},
						Resources: []string{"pods"},
					},
				},
			},
			expectSameAsOld: true,
			expectSameAsNew: true,
		},
		{
			name: "change rules when aggregation rule nil",
			oldClusterRole: &rbac.ClusterRole{
				Rules: []rbac.PolicyRule{
					{
						Verbs:     []string{"get"},
						APIGroups: []string{"*"},
						Resources: []string{"pods"},
					},
				},
			},
			newClusterRole: &rbac.ClusterRole{
				Rules: []rbac.PolicyRule{},
			},
			expectSameAsOld: false,
			expectSameAsNew: true,
		},
		{
			name: "no change rules when new rules is nil",
			oldClusterRole: &rbac.ClusterRole{
				AggregationRule: &rbac.AggregationRule{
					ClusterRoleSelectors: []metav1.LabelSelector{
						{
							MatchLabels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
				Rules: []rbac.PolicyRule{
					{
						Verbs:     []string{"get"},
						APIGroups: []string{"*"},
						Resources: []string{"pods"},
					},
				},
			},
			newClusterRole: &rbac.ClusterRole{
				AggregationRule: &rbac.AggregationRule{
					ClusterRoleSelectors: []metav1.LabelSelector{
						{
							MatchLabels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
				Rules: nil,
			},
			expectSameAsOld: true,
			expectSameAsNew: false,
		},
		{
			name: "change rules when new rules is defined with 0 item",
			oldClusterRole: &rbac.ClusterRole{
				Rules: []rbac.PolicyRule{
					{
						Verbs:     []string{"get"},
						APIGroups: []string{"*"},
						Resources: []string{"pods"},
					},
				},
			},
			newClusterRole: &rbac.ClusterRole{
				Rules: []rbac.PolicyRule{},
			},
			expectSameAsOld: false,
			expectSameAsNew: true,
		},
		{
			name: "change rules when new rules is defined with items",
			oldClusterRole: &rbac.ClusterRole{
				Rules: []rbac.PolicyRule{
					{
						Verbs:     []string{"get"},
						APIGroups: []string{"*"},
						Resources: []string{"pods"},
					},
				},
			},
			newClusterRole: &rbac.ClusterRole{
				Rules: []rbac.PolicyRule{
					{
						Verbs:     []string{"get"},
						APIGroups: []string{"*"},
						Resources: []string{"pods"},
					},
					{
						Verbs:     []string{"list"},
						APIGroups: []string{"*"},
						Resources: []string{"pods"},
					},
				},
			},
			expectSameAsOld: false,
			expectSameAsNew: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			originOld, originNew := tc.oldClusterRole.DeepCopy(), tc.newClusterRole.DeepCopy()
			s.updateRulesForAggregation(tc.oldClusterRole, tc.newClusterRole)

			if tc.expectSameAsOld && !apiequality.Semantic.DeepEqual(originOld.Rules, tc.newClusterRole.Rules) {
				t.Errorf("Expected same as old[%+v], but got[%+v]", originOld.Rules, tc.newClusterRole.Rules)
			}

			if tc.expectSameAsNew && !apiequality.Semantic.DeepEqual(originNew.Rules, tc.newClusterRole.Rules) {
				t.Errorf("Expected same as new[%+v], but got [%v]", originNew.Rules, tc.newClusterRole.Rules)
			}
		})
	}
}
