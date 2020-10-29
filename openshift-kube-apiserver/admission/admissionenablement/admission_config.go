package admissionenablement

import (
	"time"

	"github.com/openshift/library-go/pkg/apiserver/admission/admissiontimeout"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/namespaceconditions"
	controlplaneapiserver "k8s.io/kubernetes/pkg/controlplane/apiserver/options"
)

const disableSCCLevelLabel = "security.openshift.io/disable-securitycontextconstraints"

var enforceSCCSelector labels.Selector

func init() {
	var err error
	enforceSCCSelector, err = labels.Parse(disableSCCLevelLabel + " != true")
	if err != nil {
		panic(err)
	}
}

func SetAdmissionDefaults(o *controlplaneapiserver.CompletedOptions, informers informers.SharedInformerFactory, kubeClient kubernetes.Interface) {
	// set up the decorators we need.  This is done late and out of order because our decorators currently require informers which are not
	// present until we start running
	namespaceLabelDecorator := namespaceconditions.NamespaceLabelConditions{
		NamespaceClient: kubeClient.CoreV1(),
		NamespaceLister: informers.Core().V1().Namespaces().Lister(),

		SkipLevelZeroNames: SkipRunLevelZeroPlugins,
		SkipLevelOneNames:  SkipRunLevelOnePlugins,
	}
	sccLabelDecorator := namespaceconditions.NewConditionalAdmissionPlugins(
		kubeClient.CoreV1(), informers.Core().V1().Namespaces().Lister(), enforceSCCSelector,
		"security.openshift.io/SecurityContextConstraint", "security.openshift.io/SCCExecRestrictions")

	o.Admission.GenericAdmission.Decorators = append(o.Admission.GenericAdmission.Decorators,
		admission.Decorators{
			// SCC can be skipped by setting a namespace label `security.openshift.io/disable-securitycontextconstraints = true`
			// This is useful for disabling SCC and using PodSecurity admission instead.
			admission.DecoratorFunc(sccLabelDecorator.WithNamespaceLabelSelector),

			admission.DecoratorFunc(namespaceLabelDecorator.WithNamespaceLabelConditions),
			admission.DecoratorFunc(admissiontimeout.AdmissionTimeout{Timeout: 13 * time.Second}.WithTimeout),
		},
	)
}
