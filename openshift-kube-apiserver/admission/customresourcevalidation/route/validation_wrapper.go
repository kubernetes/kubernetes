package route

import (
	"fmt"

	routev1 "github.com/openshift/api/route/v1"
	"github.com/openshift/library-go/pkg/apiserver/admission/admissionrestconfig"
	authorizationv1client "k8s.io/client-go/kubernetes/typed/authorization/v1"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"

	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/client-go/rest"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation"
)

type validateCustomResourceWithClient struct {
	admission.ValidationInterface

	secretsGetter             corev1client.SecretsGetter
	sarGetter                 authorizationv1client.SubjectAccessReviewsGetter
	routeValidationOptsGetter RouteValidationOptionGetter
}

func NewValidateRoute() (admission.Interface, error) {
	ret := &validateCustomResourceWithClient{}

	delegate, err := customresourcevalidation.NewValidator(
		map[schema.GroupResource]bool{
			routev1.GroupVersion.WithResource("routes").GroupResource(): true,
		},
		map[schema.GroupVersionKind]customresourcevalidation.ObjectValidator{
			routev1.GroupVersion.WithKind("Route"): routeV1{
				secretsGetter:             ret.getSecretsGetter,
				sarGetter:                 ret.getSubjectAccessReviewsGetter,
				routeValidationOptsGetter: ret.getRouteValidationOptions,
			},
		})
	if err != nil {
		return nil, err
	}
	ret.ValidationInterface = delegate

	return ret, nil
}

var _ admissionrestconfig.WantsRESTClientConfig = &validateCustomResourceWithClient{}

func (a *validateCustomResourceWithClient) getSecretsGetter() corev1client.SecretsGetter {
	return a.secretsGetter
}

func (a *validateCustomResourceWithClient) getSubjectAccessReviewsGetter() authorizationv1client.SubjectAccessReviewsGetter {
	return a.sarGetter
}

func (a *validateCustomResourceWithClient) getRouteValidationOptions() RouteValidationOptionGetter {
	return a.routeValidationOptsGetter
}

func (a *validateCustomResourceWithClient) SetRESTClientConfig(restClientConfig rest.Config) {
	var err error

	a.secretsGetter, err = corev1client.NewForConfig(&restClientConfig)
	if err != nil {
		utilruntime.HandleError(err)
		return
	}

	a.sarGetter, err = authorizationv1client.NewForConfig(&restClientConfig)
	if err != nil {
		utilruntime.HandleError(err)
		return
	}

	a.routeValidationOptsGetter = NewRouteValidationOpts()
}

func (a *validateCustomResourceWithClient) ValidateInitialization() error {
	if a.secretsGetter == nil {
		return fmt.Errorf("%s needs a secretsGetter", PluginName)
	}
	if a.sarGetter == nil {
		return fmt.Errorf("%s needs a subjectAccessReviewsGetter", PluginName)
	}
	if a.routeValidationOptsGetter == nil {
		return fmt.Errorf("%s needs a routeValidationOptsGetter", PluginName)
	}

	return nil
}
