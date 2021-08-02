package admissionregistrationtesting

import (
	"fmt"

	"k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission"
)

func AdmissionRegistrationTest(registeredAdmission *admission.Plugins, orderedAdmissionPlugins []string, defaultOffPlugins sets.String) error {
	errs := []error{}
	registeredPlugins := sets.NewString(registeredAdmission.Registered()...)
	orderedAdmissionPluginsSet := sets.NewString(orderedAdmissionPlugins...)

	// make sure that all orderedAdmissionPlugins are registered
	if diff := orderedAdmissionPluginsSet.Difference(registeredPlugins); len(diff) > 0 {
		errs = append(errs, fmt.Errorf("registered plugins missing admission plugins:  %v", diff.List()))
	}
	if diff := defaultOffPlugins.Difference(orderedAdmissionPluginsSet); len(diff) > 0 {
		errs = append(errs, fmt.Errorf("ordered admission plugins missing defaultOff plugins: %v", diff.List()))
	}

	return errors.NewAggregate(errs)
}
