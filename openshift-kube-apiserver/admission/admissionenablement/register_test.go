package admissionenablement

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/kubernetes/pkg/kubeapiserver/options"

	"github.com/openshift/library-go/pkg/apiserver/admission/admissionregistrationtesting"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/customresourcevalidationregistration"
)

func TestAdmissionRegistration(t *testing.T) {
	orderedAdmissionChain := NewOrderedKubeAdmissionPlugins(options.AllOrderedPlugins)
	defaultOffPlugins := NewDefaultOffPluginsFunc(options.DefaultOffAdmissionPlugins())()
	registerAllAdmissionPlugins := func(plugins *admission.Plugins) {
		genericapiserver.RegisterAllAdmissionPlugins(plugins)
		options.RegisterAllAdmissionPlugins(plugins)
		RegisterOpenshiftKubeAdmissionPlugins(plugins)
		customresourcevalidationregistration.RegisterCustomResourceValidation(plugins)
	}
	plugins := admission.NewPlugins()
	registerAllAdmissionPlugins(plugins)

	err := admissionregistrationtesting.AdmissionRegistrationTest(plugins, orderedAdmissionChain, sets.Set[string](defaultOffPlugins))
	if err != nil {
		t.Fatal(err)
	}
}

// TestResourceQuotaBeforeClusterResourceQuota simply test wheather ResourceQuota plugin is before ClusterResourceQuota plugin
func TestResourceQuotaBeforeClusterResourceQuota(t *testing.T) {
	orderedAdmissionChain := NewOrderedKubeAdmissionPlugins(options.AllOrderedPlugins)

	expectedOrderedAdmissionSubChain := []string{"ResourceQuota", "quota.openshift.io/ClusterResourceQuota", "AlwaysDeny"}
	actualOrderedAdmissionChain := extractSubChain(orderedAdmissionChain, expectedOrderedAdmissionSubChain[0])

	if !reflect.DeepEqual(actualOrderedAdmissionChain, expectedOrderedAdmissionSubChain) {
		t.Fatalf("expected %v, got %v ", expectedOrderedAdmissionSubChain, actualOrderedAdmissionChain)
	}
}

func extractSubChain(admissionChain []string, takeFrom string) []string {
	indexOfTake := 0
	for index, admission := range admissionChain {
		if admission == takeFrom {
			indexOfTake = index
			break
		}
	}
	return admissionChain[indexOfTake:]
}
