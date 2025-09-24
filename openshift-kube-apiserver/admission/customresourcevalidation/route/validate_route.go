package route

import (
	"context"
	"fmt"
	"io"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
	authorizationv1client "k8s.io/client-go/kubernetes/typed/authorization/v1"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"

	routev1 "github.com/openshift/api/route/v1"
	routevalidation "github.com/openshift/library-go/pkg/route/validation"
)

const PluginName = "route.openshift.io/ValidateRoute"

func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewValidateRoute()
	})
}

func toRoute(uncastObj runtime.Object) (*routev1.Route, field.ErrorList) {
	if uncastObj == nil {
		return nil, nil
	}

	obj, ok := uncastObj.(*routev1.Route)
	if !ok {
		return nil, field.ErrorList{
			field.NotSupported(field.NewPath("kind"), fmt.Sprintf("%T", uncastObj), []string{"Route"}),
			field.NotSupported(field.NewPath("apiVersion"), fmt.Sprintf("%T", uncastObj), []string{routev1.GroupVersion.String()}),
		}
	}

	return obj, nil
}

type routeV1 struct {
	secretsGetter             func() corev1client.SecretsGetter
	sarGetter                 func() authorizationv1client.SubjectAccessReviewsGetter
	routeValidationOptsGetter func() RouteValidationOptionGetter
}

func (r routeV1) ValidateCreate(ctx context.Context, obj runtime.Object) field.ErrorList {
	routeObj, errs := toRoute(obj)
	if len(errs) > 0 {
		return errs
	}

	return routevalidation.ValidateRoute(ctx, routeObj, r.sarGetter().SubjectAccessReviews(), r.secretsGetter(), r.routeValidationOptsGetter().GetValidationOptions())
}

func (r routeV1) ValidateUpdate(ctx context.Context, obj runtime.Object, oldObj runtime.Object) field.ErrorList {
	routeObj, errs := toRoute(obj)
	if len(errs) > 0 {
		return errs
	}

	routeOldObj, errs := toRoute(oldObj)
	if len(errs) > 0 {
		return errs
	}

	return routevalidation.ValidateRouteUpdate(ctx, routeObj, routeOldObj, r.sarGetter().SubjectAccessReviews(), r.secretsGetter(), r.routeValidationOptsGetter().GetValidationOptions())
}

func (routeV1) ValidateStatusUpdate(_ context.Context, obj runtime.Object, oldObj runtime.Object) field.ErrorList {
	routeObj, errs := toRoute(obj)
	if len(errs) > 0 {
		return errs
	}

	routeOldObj, errs := toRoute(oldObj)
	if len(errs) > 0 {
		return errs
	}

	return routevalidation.ValidateRouteStatusUpdate(routeObj, routeOldObj)
}
