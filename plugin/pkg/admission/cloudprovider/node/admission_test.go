package node

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	utiltaints "k8s.io/kubernetes/pkg/util/taints"
)

func TestSetsCloudProviderLabel(t *testing.T) {
	nodeHandler := NewCloudProviderNodeTaint()
	handler := admission.NewChainHandler(nodeHandler)

	node := api.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "node1", Namespace: "default", Annotations: map[string]string{}},
		Spec:       api.NodeSpec{},
	}

	err := handler.Admit(admission.NewAttributesRecord(&node, nil, api.Kind("Node").WithVersion("version"), node.Namespace, node.Name, api.Resource("nodes").WithVersion("version"), "", admission.Create, nil))
	if err != nil {
		t.Errorf("Unexpected error returned from admission handler (on node): %v", err)
	}

	nodeTaints, err := v1.GetTaintsFromNodeAnnotations(node.Annotations)
	if err != nil {
		t.Errorf("Error getting taints from node annotations %v", err)
	}

	foundTaint := false
	taint, err := utiltaints.ParseTaint(CloudProviderTaint)
	if err != nil {
		t.Errorf("Error parsing Node taint, err :%v", err)
	}
	for _, t := range nodeTaints {
		if taint.MatchTaint(t) {
			foundTaint = true
		}
	}

	if !foundTaint {
		t.Errorf("CloudProviderNodeTaint admission controller did not set the expected taint")
	}

	err = handler.Admit(admission.NewAttributesRecord(&node, nil, api.Kind("Node").WithVersion("version"), node.Namespace, node.Name, api.Resource("nodes").WithVersion("version"), "", admission.Delete, nil))
	if err != nil {
		t.Errorf("Unexpected error returned from admission handler (on node): %v", err)
	}

}
