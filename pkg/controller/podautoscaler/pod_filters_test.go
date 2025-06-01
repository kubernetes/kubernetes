package podautoscaler

import (
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	v1 "k8s.io/api/core/v1"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/fake"
)

// Mock RESTMapper for testing
type fakeRESTMapper struct {
	apimeta.RESTMapper
}

func TestNewPodFilter(t *testing.T) {
	opts := FilterOptions{
		ScaleTargetRef: &autoscalingv2.CrossVersionObjectReference{},
	}

	// Test OwnerReferences strategy
	filter := NewPodFilter(string(autoscalingv2.OwnerReferences), opts)
	if _, ok := filter.(*OwnerReferencesFilter); !ok {
		t.Errorf("Expected OwnerReferencesFilter, got %T", filter)
	}

	// Test default strategy (should be LabelSelector)
	filter = NewPodFilter("some-other-strategy", opts)
	if _, ok := filter.(*LabelSelectorFilter); !ok {
		t.Errorf("Expected LabelSelectorFilter, got %T", filter)
	}
}

func TestLabelSelectorFilter(t *testing.T) {
	pods := []*v1.Pod{
		createTestPod("pod1", "test-namespace", nil),
		createTestPod("pod2", "test-namespace", nil),
	}

	filter := &LabelSelectorFilter{}
	filtered, unfiltered, err := filter.Filter(pods)

	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}

	if len(filtered) != 2 {
		t.Errorf("Expected 2 filtered pods, got %d", len(filtered))
	}

	if len(unfiltered) != 0 {
		t.Errorf("Expected 0 unfiltered pods, got %d", len(unfiltered))
	}

	// Test WithClient and WithRESTMapper to ensure they're no-ops
	clientFilter := filter.WithClient(nil)
	if clientFilter != filter {
		t.Errorf("WithClient should return the same filter instance")
	}

	mapperFilter := filter.WithRESTMapper(nil)
	if mapperFilter != filter {
		t.Errorf("WithRESTMapper should return the same filter instance")
	}

	// Test Name method
	if filter.Name() != "LabelSelector" {
		t.Errorf("Expected Name() to return 'LabelSelector', got '%s'", filter.Name())
	}
}

func TestOwnerReferencesFilter_MissingDependencies(t *testing.T) {
	filter := &OwnerReferencesFilter{
		filterOptions: FilterOptions{
			ScaleTargetRef: &autoscalingv2.CrossVersionObjectReference{
				Kind: "Deployment",
				Name: "test-deployment",
			},
		},
	}

	pods := []*v1.Pod{createTestPod("pod1", "test-namespace", nil)}

	// Test missing client
	_, _, err := filter.Filter(pods)
	if err == nil || err.Error() != "apps/v1 client is required for OwnerReferencesFilter" {
		t.Errorf("Expected error about missing client, got: %v", err)
	}

	// Test with client but missing RESTMapper
	fakeClient := fake.NewSimpleClientset().AppsV1()
	filter = filter.WithClient(fakeClient).(*OwnerReferencesFilter)

	_, _, err = filter.Filter(pods)
	if err == nil || err.Error() != "RESTMapper is required for OwnerReferencesFilter" {
		t.Errorf("Expected error about missing RESTMapper, got: %v", err)
	}

	// Test with both but missing ScaleTargetRef
	filter = &OwnerReferencesFilter{
		filterOptions: FilterOptions{},
		Client:        fakeClient,
		RESTMapper:    &fakeRESTMapper{},
	}

	_, _, err = filter.Filter(pods)
	if err == nil || err.Error() != "ScaleTargetRef is required for OwnerReferencesFilter" {
		t.Errorf("Expected error about missing ScaleTargetRef, got: %v", err)
	}

	// Test empty pod list
	filter = &OwnerReferencesFilter{
		filterOptions: FilterOptions{
			ScaleTargetRef: &autoscalingv2.CrossVersionObjectReference{},
		},
		Client:     fakeClient,
		RESTMapper: &fakeRESTMapper{},
	}

	filtered, unfiltered, err := filter.Filter([]*v1.Pod{})
	if err != nil {
		t.Errorf("Expected no error for empty pod list, got: %v", err)
	}
	if len(filtered) != 0 {
		t.Errorf("Expected empty filtered list, got %d pods", len(filtered))
	}
	if unfiltered != nil {
		t.Errorf("Expected nil unfiltered list, got list with %d pods", len(unfiltered))
	}
}

func TestOwnerReferencesFilter_GenericResource(t *testing.T) {
	customKind := "CustomResource"
	customName := "test-custom"

	ownedPod := createTestPod("owned-pod", "test-namespace", []metav1.OwnerReference{
		{
			Kind: customKind,
			Name: customName,
			UID:  "custom-uid",
		},
	})

	unownedPod := createTestPod("unowned-pod", "test-namespace", []metav1.OwnerReference{
		{
			Kind: "OtherKind",
			Name: "other-name",
			UID:  "other-uid",
		},
	})

	ownedPods := make(map[types.UID]bool)

	filter := &OwnerReferencesFilter{}
	filter.handleGenericResource(customKind, customName, []*v1.Pod{ownedPod, unownedPod}, ownedPods)

	if !ownedPods[ownedPod.UID] {
		t.Errorf("Expected owned pod to be marked as owned")
	}

	if ownedPods[unownedPod.UID] {
		t.Errorf("Expected unowned pod not to be marked as owned")
	}
}

func TestOwnerReferencesFilter_Deployment(t *testing.T) {

	deploymentUID := types.UID("deployment-uid")
	rsUID := types.UID("replicaset-uid")

	deployment := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-deployment",
			Namespace: "test-namespace",
			UID:       deploymentUID,
		},
		Spec: appsv1.DeploymentSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"app": "test"},
			},
		},
	}

	replicaSet := &appsv1.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-rs",
			Namespace: "test-namespace",
			UID:       rsUID,
			OwnerReferences: []metav1.OwnerReference{
				{
					Kind: "Deployment",
					Name: "test-deployment",
					UID:  deploymentUID,
				},
			},
			Labels: map[string]string{"app": "test"},
		},
	}

	client := fake.NewSimpleClientset(deployment, replicaSet).AppsV1()

	ownedPod := createTestPod("owned-pod", "test-namespace", []metav1.OwnerReference{
		{
			Kind: "ReplicaSet",
			Name: "test-rs",
			UID:  rsUID,
		},
	})

	unownedPod := createTestPod("unowned-pod", "test-namespace", []metav1.OwnerReference{
		{
			Kind: "ReplicaSet",
			Name: "other-rs",
			UID:  "other-uid",
		},
	})

	filter := &OwnerReferencesFilter{
		filterOptions: FilterOptions{
			ScaleTargetRef: &autoscalingv2.CrossVersionObjectReference{
				Kind: "Deployment",
				Name: "test-deployment",
			},
		},
		Client:     client,
		RESTMapper: &fakeRESTMapper{},
	}

	ownedPods := make(map[types.UID]bool)
	err := filter.handleDeployment("test-namespace", "test-deployment", []*v1.Pod{ownedPod, unownedPod}, ownedPods)

	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if !ownedPods[ownedPod.UID] {
		t.Errorf("Expected owned pod to be marked as owned")
	}

	if ownedPods[unownedPod.UID] {
		t.Errorf("Expected unowned pod not to be marked as owned")
	}
}

// Helper functions to create test objects
func createTestPod(name, namespace string, ownerRefs []metav1.OwnerReference) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            name,
			Namespace:       namespace,
			UID:             types.UID(name + "-uid"),
			OwnerReferences: ownerRefs,
		},
	}
}
