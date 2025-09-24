package route

import (
	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"

	openshiftfeatures "github.com/openshift/api/features"
	routecommon "github.com/openshift/library-go/pkg/route"
)

type RouteValidationOptionGetter interface {
	GetValidationOptions() routecommon.RouteValidationOptions
}

type RouteValidationOpts struct {
	opts routecommon.RouteValidationOptions
}

var _ RouteValidationOptionGetter = &RouteValidationOpts{}

func NewRouteValidationOpts() *RouteValidationOpts {
	return &RouteValidationOpts{
		opts: routecommon.RouteValidationOptions{
			AllowExternalCertificates: feature.DefaultMutableFeatureGate.Enabled(featuregate.Feature(openshiftfeatures.FeatureGateRouteExternalCertificate)),
		},
	}
}

func (o *RouteValidationOpts) GetValidationOptions() routecommon.RouteValidationOptions {
	return o.opts
}
