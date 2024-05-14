package validation

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	authorizationv1 "k8s.io/api/authorization/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	kvalidation "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/authentication/user"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"

	routev1 "github.com/openshift/api/route/v1"
	"github.com/openshift/library-go/pkg/authorization/authorizationutil"
	routecommon "github.com/openshift/library-go/pkg/route"
)

const (
	// maxHeaderNameSize is the maximum allowed length of an HTTP header
	// name.
	maxHeaderNameSize = 255
	// maxHeaderValueSize is the maximum allowed length of an HTTP header
	// value.
	maxHeaderValueSize = 16384
	// maxResponseHeaderList is the maximum allowed number of HTTP response
	// header actions.
	maxResponseHeaderList = 20
	// maxRequestHeaderList is the maximum allowed number of HTTP request
	// header actions.
	maxRequestHeaderList = 20
	// permittedHeaderNameErrorMessage is the API validation message for an
	// invalid HTTP header name.
	permittedHeaderNameErrorMessage = "name must be a valid HTTP header name as defined in RFC 2616 section 4.2"
	// permittedHeaderValueTemplate is used in the definitions of
	// permittedRequestHeaderValueRE and permittedResponseHeaderValueRE.
	// Any changes made to these regex patterns must be reflected in the
	// corresponding regexps in
	// https://github.com/openshift/api/blob/master/route/v1/types.go and
	// https://github.com/openshift/api/blob/master/operator/v1/types_ingress.go
	// for the Route.spec.httpHeaders.actions[*].response,
	// Route.spec.httpHeaders.actions[*].request,
	// IngressController.spec.httpHeaders.actions[*].response, and
	// IngressController.spec.httpHeaders.actions[*].request fields for the
	// benefit of client-side validation.
	permittedHeaderValueTemplate = `^(?:%(?:%|(?:\{[-+]?[QXE](?:,[-+]?[QXE])*\})?\[(?:XYZ\.hdr\([0-9A-Za-z-]+\)|ssl_c_der)(?:,(?:lower|base64))*\])|[^%[:cntrl:]])+$`
	// permittedRequestHeaderValueErrorMessage is the API validation message
	// for an invalid HTTP request header value.
	permittedRequestHeaderValueErrorMessage = "Either header value provided is not in correct format or the converter specified is not allowed. The dynamic header value  may use HAProxy's %[] syntax and otherwise must be a valid HTTP header value as defined in https://datatracker.ietf.org/doc/html/rfc7230#section-3.2 Sample fetchers allowed are req.hdr, ssl_c_der. Converters allowed are lower, base64."
	// permittedResponseHeaderValueErrorMessage is the API validation
	// message for an invalid HTTP response header value.
	permittedResponseHeaderValueErrorMessage = "Either header value provided is not in correct format or the converter specified is not allowed. The dynamic header value  may use HAProxy's %[] syntax and otherwise must be a valid HTTP header value as defined in https://datatracker.ietf.org/doc/html/rfc7230#section-3.2 Sample fetchers allowed are res.hdr, ssl_c_der. Converters allowed are lower, base64."
	// routerServiceAccount is used to validate RBAC permissions for externalCertificate
	routerServiceAccount = "system:serviceaccount:openshift-ingress:router"
)

var (
	// validateRouteName is a ValidateNameFunc for validating a route name.
	validateRouteName = apimachineryvalidation.NameIsDNSSubdomain
	// permittedHeaderNameRE is a compiled regexp for validating an HTTP
	// header name.
	permittedHeaderNameRE = regexp.MustCompile("^[-!#$%&'*+.0-9A-Z^_`a-z|~]+$")
	// permittedRequestHeaderValueRE is a compiled regexp for validating an
	// HTTP request header value.
	permittedRequestHeaderValueRE = regexp.MustCompile(strings.Replace(permittedHeaderValueTemplate, "XYZ", "req", 1))
	// permittedResponseHeaderValueRE is a compiled regexp for validating an
	// HTTP response header value.
	permittedResponseHeaderValueRE = regexp.MustCompile(strings.Replace(permittedHeaderValueTemplate, "XYZ", "res", 1))
)

func ValidateRoute(ctx context.Context, route *routev1.Route, sarCreator routecommon.SubjectAccessReviewCreator, secretsGetter corev1client.SecretsGetter, opts routecommon.RouteValidationOptions) field.ErrorList {
	return validateRoute(ctx, route, true, sarCreator, secretsGetter, opts)
}

// validLabels - used in the ValidateRouteUpdate function to check if "older" routes conform to DNS1123Labels or not
func validLabels(host string) bool {
	if len(host) == 0 {
		return true
	}
	return checkLabelSegments(host)
}

// checkLabelSegments - function that checks if hostname labels conform to DNS1123Labels
func checkLabelSegments(host string) bool {
	segments := strings.Split(host, ".")
	for _, s := range segments {
		errs := kvalidation.IsDNS1123Label(s)
		if len(errs) > 0 {
			return false
		}
	}
	return true
}

// validateRoute - private function to validate route
func validateRoute(ctx context.Context, route *routev1.Route, checkHostname bool, sarc routecommon.SubjectAccessReviewCreator, secrets corev1client.SecretsGetter, opts routecommon.RouteValidationOptions) field.ErrorList {
	//ensure meta is set properly
	result := validateObjectMeta(&route.ObjectMeta, true, validateRouteName, field.NewPath("metadata"))

	specPath := field.NewPath("spec")

	//host is not required but if it is set ensure it meets DNS requirements
	if len(route.Spec.Host) > 0 {
		if len(kvalidation.IsDNS1123Subdomain(route.Spec.Host)) != 0 {
			result = append(result, field.Invalid(specPath.Child("host"), route.Spec.Host, "host must conform to DNS 952 subdomain conventions"))
		}

		// Check the hostname only if the old route did not have an invalid DNS1123Label
		// and the new route cares about DNS compliant labels.
		if checkHostname && route.Annotations[routev1.AllowNonDNSCompliantHostAnnotation] != "true" {
			segments := strings.Split(route.Spec.Host, ".")
			for _, s := range segments {
				errs := kvalidation.IsDNS1123Label(s)
				for _, e := range errs {
					result = append(result, field.Invalid(specPath.Child("host"), route.Spec.Host, e))
				}
			}
		}
	}

	if len(route.Spec.Subdomain) > 0 {
		// Subdomain is not lenient because it was never used outside of
		// routes.
		//
		// TODO: Use ValidateSubdomain from library-go.
		if len(route.Spec.Subdomain) > kvalidation.DNS1123SubdomainMaxLength {
			result = append(result, field.Invalid(field.NewPath("spec.subdomain"), route.Spec.Subdomain, kvalidation.MaxLenError(kvalidation.DNS1123SubdomainMaxLength)))
		}
		for _, label := range strings.Split(route.Spec.Subdomain, ".") {
			if errs := kvalidation.IsDNS1123Label(label); len(errs) > 0 {
				result = append(result, field.Invalid(field.NewPath("spec.subdomain"), label, strings.Join(errs, ", ")))
			}
		}
	}

	if err := validateWildcardPolicy(route.Spec.Host, route.Spec.WildcardPolicy, specPath.Child("wildcardPolicy")); err != nil {
		result = append(result, err)
	}

	if route.Spec.HTTPHeaders != nil {
		if len(route.Spec.HTTPHeaders.Actions.Response) != 0 || len(route.Spec.HTTPHeaders.Actions.Request) != 0 {
			if route.Spec.TLS != nil && route.Spec.TLS.Termination == routev1.TLSTerminationPassthrough {
				result = append(result, field.Invalid(field.NewPath("spec", "tls", "termination"), route.Spec.TLS.Termination, "only edge and re-encrypt routes are supported for providing customized headers."))
			}
		}
		actionsPath := field.NewPath("spec", "httpHeaders", "actions")
		if len(route.Spec.HTTPHeaders.Actions.Response) > maxResponseHeaderList {
			result = append(result, field.Invalid(actionsPath.Child("response"), route.Spec.HTTPHeaders.Actions.Response, fmt.Sprintf("response headers list can't exceed %d items", maxResponseHeaderList)))
		} else {
			result = append(result, validateHeaders(actionsPath.Child("response"), route.Spec.HTTPHeaders.Actions.Response, permittedResponseHeaderValueRE, permittedResponseHeaderValueErrorMessage)...)
		}

		if len(route.Spec.HTTPHeaders.Actions.Request) > maxRequestHeaderList {
			result = append(result, field.Invalid(actionsPath.Child("request"), route.Spec.HTTPHeaders.Actions.Request, fmt.Sprintf("request headers list can't exceed %d items", maxRequestHeaderList)))
		} else {
			result = append(result, validateHeaders(actionsPath.Child("request"), route.Spec.HTTPHeaders.Actions.Request, permittedRequestHeaderValueRE, permittedRequestHeaderValueErrorMessage)...)
		}
	}

	if len(route.Spec.Path) > 0 && !strings.HasPrefix(route.Spec.Path, "/") {
		result = append(result, field.Invalid(specPath.Child("path"), route.Spec.Path, "path must begin with /"))
	}

	if len(route.Spec.Path) > 0 && route.Spec.TLS != nil &&
		route.Spec.TLS.Termination == routev1.TLSTerminationPassthrough {
		result = append(result, field.Invalid(specPath.Child("path"), route.Spec.Path, "passthrough termination does not support paths"))
	}

	if len(route.Spec.To.Name) == 0 {
		result = append(result, field.Required(specPath.Child("to", "name"), ""))
	}
	if route.Spec.To.Kind != "Service" {
		result = append(result, field.Invalid(specPath.Child("to", "kind"), route.Spec.To.Kind, "must reference a Service"))
	}
	if route.Spec.To.Weight != nil && (*route.Spec.To.Weight < 0 || *route.Spec.To.Weight > 256) {
		result = append(result, field.Invalid(specPath.Child("to", "weight"), route.Spec.To.Weight, "weight must be an integer between 0 and 256"))
	}

	backendPath := specPath.Child("alternateBackends")
	if len(route.Spec.AlternateBackends) > 3 {
		result = append(result, field.Required(backendPath, "cannot specify more than 3 alternate backends"))
	}
	for i, svc := range route.Spec.AlternateBackends {
		if len(svc.Name) == 0 {
			result = append(result, field.Required(backendPath.Index(i).Child("name"), ""))
		}
		if svc.Kind != "Service" {
			result = append(result, field.Invalid(backendPath.Index(i).Child("kind"), svc.Kind, "must reference a Service"))
		}
		if svc.Weight != nil && (*svc.Weight < 0 || *svc.Weight > 256) {
			result = append(result, field.Invalid(backendPath.Index(i).Child("weight"), svc.Weight, "weight must be an integer between 0 and 256"))
		}
	}

	if route.Spec.Port != nil {
		switch target := route.Spec.Port.TargetPort; {
		case target.Type == intstr.Int && target.IntVal == 0,
			target.Type == intstr.String && len(target.StrVal) == 0:
			result = append(result, field.Required(specPath.Child("port", "targetPort"), ""))
		}
	}

	if errs := validateTLS(ctx, route, specPath.Child("tls"), sarc, secrets, opts); len(errs) != 0 {
		result = append(result, errs...)
	}

	return result
}

func ValidateRouteUpdate(ctx context.Context, route *routev1.Route, older *routev1.Route, sarc routecommon.SubjectAccessReviewCreator, secrets corev1client.SecretsGetter, opts routecommon.RouteValidationOptions) field.ErrorList {
	allErrs := validateObjectMetaUpdate(&route.ObjectMeta, &older.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, apimachineryvalidation.ValidateImmutableField(route.Spec.WildcardPolicy, older.Spec.WildcardPolicy, field.NewPath("spec", "wildcardPolicy"))...)
	hostnameUpdated := route.Spec.Host != older.Spec.Host
	allErrs = append(allErrs, validateRoute(ctx, route, hostnameUpdated && validLabels(older.Spec.Host), sarc, secrets, opts)...)
	return allErrs
}

// ValidateRouteStatusUpdate validates status updates for routes.
//
// Note that this function shouldn't call ValidateRouteUpdate, otherwise
// we are risking to break existing routes.
func ValidateRouteStatusUpdate(route *routev1.Route, older *routev1.Route) field.ErrorList {
	allErrs := validateObjectMetaUpdate(&route.ObjectMeta, &older.ObjectMeta, field.NewPath("metadata"))

	// TODO: validate route status
	return allErrs
}

// validateTLS tests fields for different types of TLS combinations are set.  Called
// by ValidateRoute.
func validateTLS(ctx context.Context, route *routev1.Route, fldPath *field.Path, sarc routecommon.SubjectAccessReviewCreator, secrets corev1client.SecretsGetter, opts routecommon.RouteValidationOptions) field.ErrorList {
	result := field.ErrorList{}
	tls := route.Spec.TLS

	// no tls config present, no need for validation
	if tls == nil {
		return nil
	}

	// in all cases certificate and externalCertificate must not be specified at the same time
	switch tls.Termination {
	// reencrypt may specify destination ca cert
	// externalCert, cert, key, cacert may not be specified because the route may be a wildcard
	case routev1.TLSTerminationReencrypt:
		if opts.AllowExternalCertificates && tls.ExternalCertificate != nil {
			if len(tls.Certificate) > 0 && len(tls.ExternalCertificate.Name) > 0 {
				result = append(result, field.Invalid(fldPath.Child("externalCertificate"), tls.ExternalCertificate.Name, "cannot specify both tls.certificate and tls.externalCertificate"))
			} else if len(tls.ExternalCertificate.Name) > 0 {
				errs := validateTLSExternalCertificate(ctx, route, fldPath.Child("externalCertificate"), sarc, secrets)
				result = append(result, errs...)
			}
		}
	//passthrough term should not specify any cert
	case routev1.TLSTerminationPassthrough:
		if len(tls.Certificate) > 0 {
			result = append(result, field.Invalid(fldPath.Child("certificate"), "redacted certificate data", "passthrough termination does not support certificates"))
		}

		if len(tls.Key) > 0 {
			result = append(result, field.Invalid(fldPath.Child("key"), "redacted key data", "passthrough termination does not support certificates"))
		}

		if opts.AllowExternalCertificates && tls.ExternalCertificate != nil {
			if len(tls.ExternalCertificate.Name) > 0 {
				result = append(result, field.Invalid(fldPath.Child("externalCertificate"), tls.ExternalCertificate.Name, "passthrough termination does not support certificates"))
			}
		}

		if len(tls.CACertificate) > 0 {
			result = append(result, field.Invalid(fldPath.Child("caCertificate"), "redacted ca certificate data", "passthrough termination does not support certificates"))
		}

		if len(tls.DestinationCACertificate) > 0 {
			result = append(result, field.Invalid(fldPath.Child("destinationCACertificate"), "redacted destination ca certificate data", "passthrough termination does not support certificates"))
		}
	// edge cert should only specify cert, key, and cacert but those certs
	// may not be specified if the route is a wildcard route
	case routev1.TLSTerminationEdge:
		if len(tls.DestinationCACertificate) > 0 {
			result = append(result, field.Invalid(fldPath.Child("destinationCACertificate"), "redacted destination ca certificate data", "edge termination does not support destination certificates"))
		}

		if opts.AllowExternalCertificates && tls.ExternalCertificate != nil {
			if len(tls.Certificate) > 0 && len(tls.ExternalCertificate.Name) > 0 {
				result = append(result, field.Invalid(fldPath.Child("externalCertificate"), tls.ExternalCertificate.Name, "cannot specify both tls.certificate and tls.externalCertificate"))
			} else if len(tls.ExternalCertificate.Name) > 0 {
				errs := validateTLSExternalCertificate(ctx, route, fldPath.Child("externalCertificate"), sarc, secrets)
				result = append(result, errs...)
			}
		}

	default:
		validValues := []string{string(routev1.TLSTerminationEdge), string(routev1.TLSTerminationPassthrough), string(routev1.TLSTerminationReencrypt)}
		result = append(result, field.NotSupported(fldPath.Child("termination"), tls.Termination, validValues))
	}

	if err := validateInsecureEdgeTerminationPolicy(tls, fldPath.Child("insecureEdgeTerminationPolicy")); err != nil {
		result = append(result, err)
	}

	return result
}

// validateTLSExternalCertificate tests different pre-conditions required for
// using externalCertificate. Called by validateTLS.
func validateTLSExternalCertificate(ctx context.Context, route *routev1.Route, fldPath *field.Path, sarc routecommon.SubjectAccessReviewCreator, secretsGetter corev1client.SecretsGetter) field.ErrorList {
	tls := route.Spec.TLS

	// user must have create and update permission on the custom-host sub-resource.
	errs := routecommon.CheckRouteCustomHostSAR(ctx, fldPath, sarc)

	// The router serviceaccount must have permission to get/list/watch the referenced secret.
	// The role and rolebinding to provide this access must be provided by the user.
	if err := authorizationutil.Authorize(sarc, &user.DefaultInfo{Name: routerServiceAccount},
		&authorizationv1.ResourceAttributes{
			Namespace: route.Namespace,
			Verb:      "get",
			Resource:  "secrets",
			Name:      tls.ExternalCertificate.Name,
		}); err != nil {
		errs = append(errs, field.Forbidden(fldPath, "router serviceaccount does not have permission to get this secret"))
	}

	if err := authorizationutil.Authorize(sarc, &user.DefaultInfo{Name: routerServiceAccount},
		&authorizationv1.ResourceAttributes{
			Namespace: route.Namespace,
			Verb:      "watch",
			Resource:  "secrets",
			Name:      tls.ExternalCertificate.Name,
		}); err != nil {
		errs = append(errs, field.Forbidden(fldPath, "router serviceaccount does not have permission to watch this secret"))
	}

	if err := authorizationutil.Authorize(sarc, &user.DefaultInfo{Name: routerServiceAccount},
		&authorizationv1.ResourceAttributes{
			Namespace: route.Namespace,
			Verb:      "list",
			Resource:  "secrets",
			Name:      tls.ExternalCertificate.Name,
		}); err != nil {
		errs = append(errs, field.Forbidden(fldPath, "router serviceaccount does not have permission to list this secret"))
	}

	// The secret should be in the same namespace as that of the route.
	secret, err := secretsGetter.Secrets(route.Namespace).Get(ctx, tls.ExternalCertificate.Name, metav1.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			return append(errs, field.NotFound(fldPath, err.Error()))
		}
		return append(errs, field.InternalError(fldPath, err))
	}

	// The secret should be of type kubernetes.io/tls
	if secret.Type != corev1.SecretTypeTLS {
		errs = append(errs, field.Invalid(fldPath, tls.ExternalCertificate.Name, fmt.Sprintf("secret of type %q required", corev1.SecretTypeTLS)))
	}

	return errs
}

// validateInsecureEdgeTerminationPolicy tests fields for different types of
// insecure options. Called by validateTLS.
func validateInsecureEdgeTerminationPolicy(tls *routev1.TLSConfig, fldPath *field.Path) *field.Error {
	// Check insecure option value if specified (empty is ok).
	if len(tls.InsecureEdgeTerminationPolicy) == 0 {
		return nil
	}

	// It is an edge-terminated or reencrypt route, check insecure option value is
	// one of None(for disable), Allow or Redirect.
	allowedValues := map[routev1.InsecureEdgeTerminationPolicyType]struct{}{
		routev1.InsecureEdgeTerminationPolicyNone:     {},
		routev1.InsecureEdgeTerminationPolicyAllow:    {},
		routev1.InsecureEdgeTerminationPolicyRedirect: {},
	}

	switch tls.Termination {
	case routev1.TLSTerminationReencrypt:
		fallthrough
	case routev1.TLSTerminationEdge:
		if _, ok := allowedValues[tls.InsecureEdgeTerminationPolicy]; !ok {
			msg := fmt.Sprintf("invalid value for InsecureEdgeTerminationPolicy option, acceptable values are %s, %s, %s, or empty", routev1.InsecureEdgeTerminationPolicyNone, routev1.InsecureEdgeTerminationPolicyAllow, routev1.InsecureEdgeTerminationPolicyRedirect)
			return field.Invalid(fldPath, tls.InsecureEdgeTerminationPolicy, msg)
		}
	case routev1.TLSTerminationPassthrough:
		if routev1.InsecureEdgeTerminationPolicyNone != tls.InsecureEdgeTerminationPolicy && routev1.InsecureEdgeTerminationPolicyRedirect != tls.InsecureEdgeTerminationPolicy {
			msg := fmt.Sprintf("invalid value for InsecureEdgeTerminationPolicy option, acceptable values are %s, %s, or empty", routev1.InsecureEdgeTerminationPolicyNone, routev1.InsecureEdgeTerminationPolicyRedirect)
			return field.Invalid(fldPath, tls.InsecureEdgeTerminationPolicy, msg)
		}
	}

	return nil
}

var (
	allowedWildcardPolicies    = []string{string(routev1.WildcardPolicyNone), string(routev1.WildcardPolicySubdomain)}
	allowedWildcardPoliciesSet = sets.New(allowedWildcardPolicies...)
)

// validateWildcardPolicy tests that the wildcard policy is either empty or one of the supported types.
func validateWildcardPolicy(host string, policy routev1.WildcardPolicyType, fldPath *field.Path) *field.Error {
	if len(policy) == 0 {
		return nil
	}

	// Check if policy is one of None or Subdomain.
	if !allowedWildcardPoliciesSet.Has(string(policy)) {
		return field.NotSupported(fldPath, policy, allowedWildcardPolicies)
	}

	if policy == routev1.WildcardPolicySubdomain && len(host) == 0 {
		return field.Invalid(fldPath, policy, "host name not specified for wildcard policy")
	}

	return nil
}

var (
	notAllowedHTTPHeaders        = []string{"strict-transport-security", "proxy", "cookie", "set-cookie"}
	notAllowedHTTPHeaderSet      = sets.New(notAllowedHTTPHeaders...)
	notAllowedHTTPHeadersMessage = fmt.Sprintf("the following headers may not be modified using this API: %v", strings.Join(notAllowedHTTPHeaders, ", "))
)

// validateHeaders verifies that the given slice of request or response headers
// is valid using the given regexp.
func validateHeaders(fldPath *field.Path, headers []routev1.RouteHTTPHeader, valueRegexpForHeaderValue *regexp.Regexp, valueErrorMessage string) field.ErrorList {
	allErrs := field.ErrorList{}
	headersMap := map[string]struct{}{}
	for i, header := range headers {
		idxPath := fldPath.Index(i)

		// Each action must specify a unique header.
		_, alreadyExists := headersMap[header.Name]
		if alreadyExists {
			err := field.Duplicate(idxPath.Child("name"), header.Name)
			allErrs = append(allErrs, err)
		}
		headersMap[header.Name] = struct{}{}

		switch nameLength := len(header.Name); {
		case nameLength == 0:
			err := field.Required(idxPath.Child("name"), "")
			allErrs = append(allErrs, err)
		case nameLength > maxHeaderNameSize:
			err := field.Invalid(idxPath.Child("name"), header.Name, fmt.Sprintf("name exceeds the maximum length, which is %d", maxHeaderNameSize))
			allErrs = append(allErrs, err)
		case notAllowedHTTPHeaderSet.Has(strings.ToLower(header.Name)):
			err := field.Forbidden(idxPath.Child("name"), notAllowedHTTPHeadersMessage)
			allErrs = append(allErrs, err)
		case !permittedHeaderNameRE.MatchString(header.Name):
			err := field.Invalid(idxPath.Child("name"), header.Name, permittedHeaderNameErrorMessage)
			allErrs = append(allErrs, err)
		}

		if header.Action.Type != routev1.Set && header.Action.Type != routev1.Delete {
			err := field.Invalid(idxPath.Child("action", "type"), header.Action.Type, fmt.Sprintf("type must be %q or %q", routev1.Set, routev1.Delete))
			allErrs = append(allErrs, err)
		}

		if header.Action.Type == routev1.Set && header.Action.Set == nil || header.Action.Type != routev1.Set && header.Action.Set != nil {
			err := field.Required(idxPath.Child("action", "set"), "set is required when type is Set, and forbidden otherwise")
			allErrs = append(allErrs, err)
		}
		if header.Action.Set != nil {
			switch valueLength := len(header.Action.Set.Value); {
			case valueLength == 0:
				err := field.Required(idxPath.Child("action", "set", "value"), "")
				allErrs = append(allErrs, err)
			case valueLength > maxHeaderValueSize:
				err := field.Invalid(idxPath.Child("action", "set", "value"), header.Action.Set.Value, fmt.Sprintf("value exceeds the maximum length, which is %d", maxHeaderValueSize))
				allErrs = append(allErrs, err)
			case !valueRegexpForHeaderValue.MatchString(header.Action.Set.Value):
				err := field.Invalid(idxPath.Child("action", "set", "value"), header.Action.Set.Value, valueErrorMessage)
				allErrs = append(allErrs, err)
			}
		}
	}
	return allErrs
}

// The special finalizer name validations were copied from k8s.io/kubernetes to eliminate that
// dependency and preserve the existing behavior.

// k8s.io/kubernetes/pkg/apis/core/validation.ValidateObjectMeta
func validateObjectMeta(meta *metav1.ObjectMeta, requiresNamespace bool, nameFn apimachineryvalidation.ValidateNameFunc, fldPath *field.Path) field.ErrorList {
	allErrs := apimachineryvalidation.ValidateObjectMeta(meta, requiresNamespace, apimachineryvalidation.ValidateNameFunc(nameFn), fldPath)
	// run additional checks for the finalizer name
	for i := range meta.Finalizers {
		allErrs = append(allErrs, validateKubeFinalizerName(string(meta.Finalizers[i]), fldPath.Child("finalizers").Index(i))...)
	}
	return allErrs
}

// k8s.io/kubernetes/pkg/apis/core/validation.ValidateObjectMetaUpdate
func validateObjectMetaUpdate(newMeta, oldMeta *metav1.ObjectMeta, fldPath *field.Path) field.ErrorList {
	allErrs := apimachineryvalidation.ValidateObjectMetaUpdate(newMeta, oldMeta, fldPath)
	// run additional checks for the finalizer name
	for i := range newMeta.Finalizers {
		allErrs = append(allErrs, validateKubeFinalizerName(string(newMeta.Finalizers[i]), fldPath.Child("finalizers").Index(i))...)
	}

	return allErrs
}

var standardFinalizers = sets.New(
	string(corev1.FinalizerKubernetes),
	metav1.FinalizerOrphanDependents,
	metav1.FinalizerDeleteDependents,
)

func validateKubeFinalizerName(stringValue string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(strings.Split(stringValue, "/")) == 1 {
		if !standardFinalizers.Has(stringValue) {
			return append(allErrs, field.Invalid(fldPath, stringValue, "name is neither a standard finalizer name nor is it fully qualified"))
		}
	}

	return allErrs
}

func Warnings(route *routev1.Route) []string {
	if len(route.Spec.Host) != 0 && len(route.Spec.Subdomain) != 0 {
		var warnings []string
		warnings = append(warnings, "spec.host is set; spec.subdomain may be ignored")
		return warnings
	}
	return nil
}
