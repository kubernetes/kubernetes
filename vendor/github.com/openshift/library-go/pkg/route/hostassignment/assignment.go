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
)

// HostGeneratedAnnotationKey is the key for an annotation set to "true" if the route's host was generated
const HostGeneratedAnnotationKey = "openshift.io/host.generated"

// Registry is an interface for performing subject access reviews
type SubjectAccessReviewCreator interface {
	Create(ctx context.Context, sar *authorizationv1.SubjectAccessReview, opts metav1.CreateOptions) (*authorizationv1.SubjectAccessReview, error)
}

type HostnameGenerator interface {
	GenerateHostname(*routev1.Route) (string, error)
}

// AllocateHost allocates a host name ONLY if the route doesn't specify a subdomain wildcard policy and
// the host name on the route is empty and an allocator is configured.
// It must first allocate the shard and may return an error if shard allocation fails.
func AllocateHost(ctx context.Context, route *routev1.Route, sarc SubjectAccessReviewCreator, routeAllocator HostnameGenerator) field.ErrorList {
	hostSet := len(route.Spec.Host) > 0
	certSet := route.Spec.TLS != nil && (len(route.Spec.TLS.CACertificate) > 0 || len(route.Spec.TLS.Certificate) > 0 || len(route.Spec.TLS.DestinationCACertificate) > 0 || len(route.Spec.TLS.Key) > 0)
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

func hasCertificateInfo(tls *routev1.TLSConfig) bool {
	if tls == nil {
		return false
	}
	return len(tls.Certificate) > 0 ||
		len(tls.Key) > 0 ||
		len(tls.CACertificate) > 0 ||
		len(tls.DestinationCACertificate) > 0
}

func certificateChangeRequiresAuth(route, older *routev1.Route) bool {
	switch {
	case route.Spec.TLS != nil && older.Spec.TLS != nil:
		a, b := route.Spec.TLS, older.Spec.TLS
		if !hasCertificateInfo(a) {
			// removing certificate info is allowed
			return false
		}
		return a.CACertificate != b.CACertificate ||
			a.Certificate != b.Certificate ||
			a.DestinationCACertificate != b.DestinationCACertificate ||
			a.Key != b.Key
	case route.Spec.TLS != nil:
		// using any default certificate is allowed
		return hasCertificateInfo(route.Spec.TLS)
	default:
		// all other cases we are not adding additional certificate info
		return false
	}
}

func ValidateHostUpdate(ctx context.Context, route, older *routev1.Route, sarc SubjectAccessReviewCreator) field.ErrorList {
	hostChanged := route.Spec.Host != older.Spec.Host
	subdomainChanged := route.Spec.Subdomain != older.Spec.Subdomain
	certChanged := certificateChangeRequiresAuth(route, older)
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
			return errs
		}
	}
	return nil
}
