package admissionenablement

import (
	"k8s.io/apimachinery/pkg/util/sets"

	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/customresourcevalidationregistration"
)

func InstallOpenShiftAdmissionPlugins(o *options.ServerRunOptions) {
	existingAdmissionOrder := o.Admission.GenericAdmission.RecommendedPluginOrder
	o.Admission.GenericAdmission.RecommendedPluginOrder = NewOrderedKubeAdmissionPlugins(existingAdmissionOrder)
	RegisterOpenshiftKubeAdmissionPlugins(o.Admission.GenericAdmission.Plugins)
	customresourcevalidationregistration.RegisterCustomResourceValidation(o.Admission.GenericAdmission.Plugins)
	existingDefaultOff := o.Admission.GenericAdmission.DefaultOffPlugins
	o.Admission.GenericAdmission.DefaultOffPlugins = sets.StringKeySet(NewDefaultOffPluginsFunc(existingDefaultOff)())
}
