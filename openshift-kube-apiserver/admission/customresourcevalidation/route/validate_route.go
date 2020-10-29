package route

import (
	"fmt"
	"io"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"

	routev1 "github.com/openshift/api/route/v1"
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
}

func (routeV1) ValidateCreate(obj runtime.Object) field.ErrorList {
	routeObj, errs := toRoute(obj)
	if len(errs) > 0 {
		return errs
	}

	return routevalidation.ValidateRoute(routeObj)
}

func (routeV1) ValidateUpdate(obj runtime.Object, oldObj runtime.Object) field.ErrorList {
	routeObj, errs := toRoute(obj)
	if len(errs) > 0 {
		return errs
	}

	routeOldObj, errs := toRoute(oldObj)
	if len(errs) > 0 {
		return errs
	}

	return routevalidation.ValidateRouteUpdate(routeObj, routeOldObj)
}

func (c routeV1) ValidateStatusUpdate(obj runtime.Object, oldObj runtime.Object) field.ErrorList {
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
