package daemon

import (
	"testing"

	projectv1 "github.com/openshift/api/project/v1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
)

func TestNamespaceNodeSelectorMatches(t *testing.T) {
	nodes := []*v1.Node{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "first",
				Labels: map[string]string{
					"alpha": "bravo",
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "second",
				Labels: map[string]string{
					"charlie": "delta",
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "third",
				Labels: map[string]string{
					"echo": "foxtrot",
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "fourth",
				Labels: map[string]string{},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "fifth",
				Labels: map[string]string{
					"charlie": "delta",
					"echo":    "foxtrot",
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "sixth",
				Labels: map[string]string{
					"alpha":   "bravo",
					"charlie": "delta",
					"echo":    "foxtrot",
				},
			},
		},
	}

	pureDefault := &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{},
	}
	all := &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				projectv1.ProjectNodeSelector: "",
			},
		},
	}
	projectSpecified := &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				projectv1.ProjectNodeSelector: "echo=foxtrot",
			},
		},
	}
	schedulerSpecified := &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				"scheduler.alpha.kubernetes.io/node-selector": "charlie=delta",
			},
		},
	}
	bothSpecified := &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				projectv1.ProjectNodeSelector:                 "echo=foxtrot",
				"scheduler.alpha.kubernetes.io/node-selector": "charlie=delta",
			},
		},
	}

	tests := []struct {
		name            string
		defaultSelector string
		namespace       *v1.Namespace
		expected        map[string]bool
	}{
		{
			name:            "pure-default",
			defaultSelector: "alpha=bravo",
			namespace:       pureDefault,
			expected: map[string]bool{
				"first": true,
				"sixth": true,
			},
		},
		{
			name:            "all",
			defaultSelector: "alpha=bravo",
			namespace:       all,
			expected: map[string]bool{
				"first":  true,
				"second": true,
				"third":  true,
				"fourth": true,
				"fifth":  true,
				"sixth":  true,
			},
		},
		{
			name:      "pure-default-without-default",
			namespace: pureDefault,
			expected: map[string]bool{
				"first":  true,
				"second": true,
				"third":  true,
				"fourth": true,
				"fifth":  true,
				"sixth":  true,
			},
		},
		{
			name:      "projectSpecified",
			namespace: projectSpecified,
			expected: map[string]bool{
				"third": true,
				"fifth": true,
				"sixth": true,
			},
		},
		{
			name:      "schedulerSpecified",
			namespace: schedulerSpecified,
			expected: map[string]bool{
				"second": true,
				"fifth":  true,
				"sixth":  true,
			},
		},
		{
			name:      "bothSpecified",
			namespace: bothSpecified,
			expected: map[string]bool{
				"second": true,
				"fifth":  true,
				"sixth":  true,
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			c := &DaemonSetsController{}
			c.openshiftDefaultNodeSelectorString = test.defaultSelector
			if len(c.openshiftDefaultNodeSelectorString) > 0 {
				var err error
				c.openshiftDefaultNodeSelector, err = labels.Parse(c.openshiftDefaultNodeSelectorString)
				if err != nil {
					t.Fatal(err)
				}
			}

			for _, node := range nodes {
				if e, a := test.expected[node.Name], c.nodeSelectorMatches(node, test.namespace); e != a {
					t.Errorf("%q expected %v, got %v", node.Name, e, a)
				}
			}
		})
	}
}
