package hostassignment

import (
	"context"
	"fmt"

	authorizationv1 "k8s.io/api/authorization/v1"
	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/endpoints/request"

	routev1 "github.com/openshift/api/route/v1"
	"github.com/openshift/library-go/pkg/authorization/authorizationutil"
	"github.com/openshift/library-go/pkg/route"
)

// HostGeneratedAnnotationKey is the key for an annotation set to "true" if the route's host was generated
const HostGeneratedAnnotationKey = "openshift.io/host.generated"

type HostnameGenerator interface {
	GenerateHostname(*routev1.Route) (string, error)
}

// AllocateHost allocates a host name ONLY if the route doesn't specify a subdomain wildcard policy and
// the host name on the route is empty and an allocator is configured.
// It must first allocate the shard and may return an error if shard allocation fails.
func AllocateHost(ctx context.Context, route *routev1.Route, sarc route.SubjectAccessReviewCreator, routeAllocator HostnameGenerator, opts route.RouteValidationOptions) field.ErrorList {
	hostSet := len(route.Spec.Host) > 0
	certSet := route.Spec.TLS != nil &&
		(len(route.Spec.TLS.CACertificate) > 0 ||
			len(route.Spec.TLS.Certificate) > 0 ||
			len(route.Spec.TLS.DestinationCACertificate) > 0 ||
			len(route.Spec.TLS.Key) > 0)

	if opts.AllowExternalCertificates && route.Spec.TLS != nil && route.Spec.TLS.ExternalCertificate != nil {
		certSet = certSet || len(route.Spec.TLS.ExternalCertificate.Name) > 0
	}

	if hostSet || certSet {
		user, ok := request.UserFrom(ctx)
		if !ok {
			return field.ErrorList{field.InternalError(field.NewPath("spec", "host"), fmt.Errorf("unable to verify host field can be set"))}
		}
		res, err := sarc.Create(
			ctx,
			authorizationutil.AddUserToSAR(
				user,
				&authorizationv1.SubjectAccessReview{
					Spec: authorizationv1.SubjectAccessReviewSpec{
						ResourceAttributes: &authorizationv1.ResourceAttributes{
							Namespace:   request.NamespaceValue(ctx),
							Verb:        "create",
							Group:       routev1.GroupName,
							Resource:    "routes",
							Subresource: "custom-host",
						},
					},
				},
			),
			metav1.CreateOptions{},
		)
		if err != nil {
			return field.ErrorList{field.InternalError(field.NewPath("spec", "host"), err)}
		}
		if !res.Status.Allowed {
			if hostSet {
				return field.ErrorList{field.Forbidden(field.NewPath("spec", "host"), "you do not have permission to set the host field of the route")}
			}
			return field.ErrorList{field.Forbidden(field.NewPath("spec", "tls"), "you do not have permission to set certificate fields on the route")}
		}
	}

	if route.Spec.WildcardPolicy == routev1.WildcardPolicySubdomain {
		// Don't allocate a host if subdomain wildcard policy.
		return nil
	}

	if len(route.Spec.Subdomain) == 0 && len(route.Spec.Host) == 0 && routeAllocator != nil {
		// TODO: this does not belong here, and should be removed
		host, err := routeAllocator.GenerateHostname(route)
		if err != nil {
			return field.ErrorList{field.InternalError(field.NewPath("spec", "host"), fmt.Errorf("allocation error: %v for route: %#v", err, route))}
		}
		route.Spec.Host = host
		if route.Annotations == nil {
			route.Annotations = map[string]string{}
		}
		route.Annotations[HostGeneratedAnnotationKey] = "true"
	}
	return nil
}

func hasCertificateInfo(tls *routev1.TLSConfig, opts route.RouteValidationOptions) bool {
	if tls == nil {
		return false
	}
	hasInfo := len(tls.Certificate) > 0 ||
		len(tls.Key) > 0 ||
		len(tls.CACertificate) > 0 ||
		len(tls.DestinationCACertificate) > 0

	if opts.AllowExternalCertificates && tls.ExternalCertificate != nil {
		hasInfo = hasInfo || len(tls.ExternalCertificate.Name) > 0
	}
	return hasInfo
}

// certificateChangeRequiresAuth determines whether changes to the TLS certificate configuration require authentication.
// Note: If either route uses externalCertificate, this function always returns true, as we cannot definitively verify if
// the content of the referenced secret has been modified. Even if the secret name remains the same,
// we must assume that the secret content is changed, necessitating authorization.
func certificateChangeRequiresAuth(route, older *routev1.Route, opts route.RouteValidationOptions) bool {
	switch {
	case route.Spec.TLS != nil && older.Spec.TLS != nil:
		a, b := route.Spec.TLS, older.Spec.TLS
		if !hasCertificateInfo(a, opts) {
			// removing certificate info is allowed
			return false
		}

		certChanged := a.CACertificate != b.CACertificate ||
			a.Certificate != b.Certificate ||
			a.DestinationCACertificate != b.DestinationCACertificate ||
			a.Key != b.Key

		if opts.AllowExternalCertificates {
			if route.Spec.TLS.ExternalCertificate != nil || older.Spec.TLS.ExternalCertificate != nil {
				certChanged = true
			}
		}

		return certChanged
	case route.Spec.TLS != nil:
		// using any default certificate is allowed
		return hasCertificateInfo(route.Spec.TLS, opts)
	default:
		// all other cases we are not adding additional certificate info
		return false
	}
}

// ValidateHostUpdate checks if the user has the correct permissions based on the updates
// done to the route object. If the route's host/subdomain has been updated it checks if
// the user has "update" permission on custom-host subresource. If only the certificate
// has changed, it checks if the user has "create" permission on the custom-host subresource.
// Caveat here is that if the route uses externalCertificate, the certChanged condition will
// always be true since we cannot verify state of external secret object.
func ValidateHostUpdate(ctx context.Context, route, older *routev1.Route, sarc route.SubjectAccessReviewCreator, opts route.RouteValidationOptions) field.ErrorList {
	hostChanged := route.Spec.Host != older.Spec.Host
	subdomainChanged := route.Spec.Subdomain != older.Spec.Subdomain
	certChanged := certificateChangeRequiresAuth(route, older, opts)
	if !hostChanged && !certChanged && !subdomainChanged {
		return nil
	}
	user, ok := request.UserFrom(ctx)
	if !ok {
		return field.ErrorList{field.InternalError(field.NewPath("spec", "host"), fmt.Errorf("unable to verify host field can be changed"))}
	}
	res, err := sarc.Create(
		ctx,
		authorizationutil.AddUserToSAR(
			user,
			&authorizationv1.SubjectAccessReview{
				Spec: authorizationv1.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationv1.ResourceAttributes{
						Namespace:   request.NamespaceValue(ctx),
						Verb:        "update",
						Group:       routev1.GroupName,
						Resource:    "routes",
						Subresource: "custom-host",
					},
				},
			},
		),
		metav1.CreateOptions{},
	)
	if err != nil {
		if subdomainChanged {
			return field.ErrorList{field.InternalError(field.NewPath("spec", "subdomain"), err)}
		}
		return field.ErrorList{field.InternalError(field.NewPath("spec", "host"), err)}
	}
	if !res.Status.Allowed {
		if hostChanged {
			return apimachineryvalidation.ValidateImmutableField(route.Spec.Host, older.Spec.Host, field.NewPath("spec", "host"))
		}
		if subdomainChanged {
			return apimachineryvalidation.ValidateImmutableField(route.Spec.Subdomain, older.Spec.Subdomain, field.NewPath("spec", "subdomain"))
		}

		// if tls is being updated without host being updated, we check if 'create' permission exists on custom-host subresource
		res, err := sarc.Create(
			ctx,
			authorizationutil.AddUserToSAR(
				user,
				&authorizationv1.SubjectAccessReview{
					Spec: authorizationv1.SubjectAccessReviewSpec{
						ResourceAttributes: &authorizationv1.ResourceAttributes{
							Namespace:   request.NamespaceValue(ctx),
							Verb:        "create",
							Group:       routev1.GroupName,
							Resource:    "routes",
							Subresource: "custom-host",
						},
					},
				},
			),
			metav1.CreateOptions{},
		)
		if err != nil {
			return field.ErrorList{field.InternalError(field.NewPath("spec", "host"), err)}
		}
		if !res.Status.Allowed {
			if route.Spec.TLS == nil || older.Spec.TLS == nil {
				return apimachineryvalidation.ValidateImmutableField(route.Spec.TLS, older.Spec.TLS, field.NewPath("spec", "tls"))
			}
			errs := apimachineryvalidation.ValidateImmutableField(route.Spec.TLS.CACertificate, older.Spec.TLS.CACertificate, field.NewPath("spec", "tls", "caCertificate"))
			errs = append(errs, apimachineryvalidation.ValidateImmutableField(route.Spec.TLS.Certificate, older.Spec.TLS.Certificate, field.NewPath("spec", "tls", "certificate"))...)
			errs = append(errs, apimachineryvalidation.ValidateImmutableField(route.Spec.TLS.DestinationCACertificate, older.Spec.TLS.DestinationCACertificate, field.NewPath("spec", "tls", "destinationCACertificate"))...)
			errs = append(errs, apimachineryvalidation.ValidateImmutableField(route.Spec.TLS.Key, older.Spec.TLS.Key, field.NewPath("spec", "tls", "key"))...)

			if opts.AllowExternalCertificates {
				if route.Spec.TLS.ExternalCertificate == nil || older.Spec.TLS.ExternalCertificate == nil {
					errs = append(errs, apimachineryvalidation.ValidateImmutableField(route.Spec.TLS.ExternalCertificate, older.Spec.TLS.ExternalCertificate, field.NewPath("spec", "tls", "externalCertificate"))...)
				} else {
					errs = append(errs, apimachineryvalidation.ValidateImmutableField(route.Spec.TLS.ExternalCertificate.Name, older.Spec.TLS.ExternalCertificate.Name, field.NewPath("spec", "tls", "externalCertificate"))...)
				}
			}
			return errs
		}
	}
	return nil
}
