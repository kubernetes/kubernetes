package admissionenablement

import (
	"time"

	"github.com/openshift/library-go/pkg/apiserver/admission/admissiontimeout"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/namespaceconditions"
)

func SetAdmissionDefaults(o *options.ServerRunOptions, informers informers.SharedInformerFactory, kubeClient kubernetes.Interface) {
	// set up the decorators we need.  This is done late and out of order because our decorators currently require informers which are not
	// present until we start running
	namespaceLabelDecorator := namespaceconditions.NamespaceLabelConditions{
		NamespaceClient: kubeClient.CoreV1(),
		NamespaceLister: informers.Core().V1().Namespaces().Lister(),

		SkipLevelZeroNames: SkipRunLevelZeroPlugins,
		SkipLevelOneNames:  SkipRunLevelOnePlugins,
	}
	o.Admission.GenericAdmission.Decorators = append(o.Admission.GenericAdmission.Decorators,
		admission.Decorators{
			admission.DecoratorFunc(namespaceLabelDecorator.WithNamespaceLabelConditions),
			admission.DecoratorFunc(admissiontimeout.AdmissionTimeout{Timeout: 13 * time.Second}.WithTimeout),
		},
	)
}
