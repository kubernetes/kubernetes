package hostassignment

import (
	"context"

	"k8s.io/apimachinery/pkg/util/validation/field"

	routev1 "github.com/openshift/api/route/v1"
	routecommon "github.com/openshift/library-go/pkg/route"
)

// ValidateHostExternalCertificate checks if the user has permissions to create and update
// custom-host subresource of routes. This check is required to be done prior to ValidateHostUpdate()
// since updating hosts while using externalCertificate is contingent on the user having both these
// permissions. The ValidateHostUpdate() cannot differentiate if the certificate has changed since
// now the certificates will be present as a secret object, due to this it proceeds with the assumption
// that the certificate has changed when the route has externalCertificate set.
// TODO: Consider merging this function into ValidateHostUpdate.
func ValidateHostExternalCertificate(ctx context.Context, new, older *routev1.Route, sarc routecommon.SubjectAccessReviewCreator, opts routecommon.RouteValidationOptions) field.ErrorList {

	if !opts.AllowExternalCertificates {
		// Return nil since the feature gate is off.
		// ValidateHostUpdate() is sufficient to validate
		// permissions.
		return nil
	}

	newTLS := new.Spec.TLS
	oldTLS := older.Spec.TLS
	if (newTLS != nil && newTLS.ExternalCertificate != nil) || (oldTLS != nil && oldTLS.ExternalCertificate != nil) {
		return routecommon.CheckRouteCustomHostSAR(ctx, field.NewPath("spec", "tls", "externalCertificate"), sarc)
	}

	return nil
}
