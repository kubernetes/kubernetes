package route

import (
	"context"
	"fmt"
	"io"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/client-go/kubernetes"
	corev1 "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/component-base/featuregate"

	configv1 "github.com/openshift/api/config/v1"
	routev1 "github.com/openshift/api/route/v1"
	"github.com/openshift/library-go/pkg/route"
	routevalidation "github.com/openshift/library-go/pkg/route/validation"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation"
)

const PluginName = "route.openshift.io/ValidateRoute"

func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return customresourcevalidation.NewValidator(
			map[schema.GroupResource]bool{
				{Group: routev1.GroupName, Resource: "routes"}: true,
			},
			map[schema.GroupVersionKind]customresourcevalidation.ObjectValidator{
				routev1.GroupVersion.WithKind("Route"): routeV1{},
			})
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
	secrets                   corev1.SecretsGetter
	sars                      route.SubjectAccessReviewCreator
	allowExternalCertificates bool
	inspectedFeatureGates     bool
}

var _ initializer.WantsExternalKubeClientSet = &routeV1{}
var _ initializer.WantsFeatures = &routeV1{}

func (r *routeV1) SetExternalKubeClientSet(k kubernetes.Interface) {
	r.secrets = k.CoreV1()
	r.sars = k.AuthorizationV1().SubjectAccessReviews()
}

func (r *routeV1) InspectFeatureGates(fgs featuregate.FeatureGate) {
	r.allowExternalCertificates = fgs.Enabled(featuregate.Feature(configv1.FeatureGateRouteExternalCertificate))
	r.inspectedFeatureGates = true
}

func (r *routeV1) ValidateInitialization() error {
	if !r.inspectedFeatureGates {
		return fmt.Errorf("did not inspect feature gates")
	}
	if r.secrets == nil {
		return fmt.Errorf("missing secrets getter")
	}
	if r.sars == nil {
		return fmt.Errorf("missing subject access review creator")
	}
	return nil
}

func (r routeV1) ValidateCreate(ctx context.Context, obj runtime.Object) field.ErrorList {
	routeObj, errs := toRoute(obj)
	if len(errs) > 0 {
		return errs
	}

	return routevalidation.ValidateRoute(ctx, routeObj, nil, r.secrets, route.RouteValidationOptions{AllowExternalCertificates: r.allowExternalCertificates})
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

	return routevalidation.ValidateRouteUpdate(ctx, routeObj, routeOldObj, r.sars, r.secrets, route.RouteValidationOptions{AllowExternalCertificates: r.allowExternalCertificates})
}

func (r routeV1) ValidateStatusUpdate(_ context.Context, obj runtime.Object, oldObj runtime.Object) field.ErrorList {
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
