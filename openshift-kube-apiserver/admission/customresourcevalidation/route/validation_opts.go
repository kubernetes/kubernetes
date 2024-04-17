package route

import (
	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"

	configv1 "github.com/openshift/api/config/v1"
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
			AllowExternalCertificates: feature.DefaultMutableFeatureGate.Enabled(featuregate.Feature(configv1.FeatureGateRouteExternalCertificate)),
		},
	}
}

func (o *RouteValidationOpts) GetValidationOptions() routecommon.RouteValidationOptions {
	return o.opts
}
