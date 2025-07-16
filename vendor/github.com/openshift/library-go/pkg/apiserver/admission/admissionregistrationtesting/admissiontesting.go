package admissionregistrationtesting

import (
	"fmt"

	"k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission"
)

func AdmissionRegistrationTest(registeredAdmission *admission.Plugins, orderedAdmissionPlugins []string, defaultOffPlugins sets.Set[string]) error {
	errs := []error{}
	registeredPlugins := sets.New(registeredAdmission.Registered()...)
	orderedAdmissionPluginsSet := sets.New(orderedAdmissionPlugins...)

	// make sure that all orderedAdmissionPlugins are registered
	if diff := orderedAdmissionPluginsSet.Difference(registeredPlugins); len(diff) > 0 {
		errs = append(errs, fmt.Errorf("registered plugins missing admission plugins:  %v", sets.List(diff)))
	}
	if diff := defaultOffPlugins.Difference(orderedAdmissionPluginsSet); len(diff) > 0 {
		errs = append(errs, fmt.Errorf("ordered admission plugins missing defaultOff plugins: %v", sets.List(diff)))
	}

	return errors.NewAggregate(errs)
}
