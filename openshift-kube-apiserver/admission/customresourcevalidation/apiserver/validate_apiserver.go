package apiserver

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	"k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation"

	configv1 "github.com/openshift/api/config/v1"
	configv1client "github.com/openshift/client-go/config/clientset/versioned/typed/config/v1"
	libgocrypto "github.com/openshift/library-go/pkg/crypto"
)

func toAPIServerV1(uncastObj runtime.Object) (*configv1.APIServer, field.ErrorList) {
	if uncastObj == nil {
		return nil, nil
	}

	errs := field.ErrorList{}

	obj, ok := uncastObj.(*configv1.APIServer)
	if !ok {
		return nil, append(errs,
			field.NotSupported(field.NewPath("kind"), fmt.Sprintf("%T", uncastObj), []string{"APIServer"}),
			field.NotSupported(field.NewPath("apiVersion"), fmt.Sprintf("%T", uncastObj), []string{"config.openshift.io/v1"}))
	}

	return obj, nil
}

type apiserverV1 struct {
	infrastructureGetter func() configv1client.InfrastructuresGetter
}

func (a apiserverV1) ValidateCreate(_ context.Context, uncastObj runtime.Object) field.ErrorList {
	obj, errs := toAPIServerV1(uncastObj)
	if len(errs) > 0 {
		return errs
	}

	errs = append(errs, validation.ValidateObjectMeta(&obj.ObjectMeta, false, customresourcevalidation.RequireNameCluster, field.NewPath("metadata"))...)
	errs = append(errs, validateAPIServerSpecCreate(obj.Spec)...)
	errs = append(errs, a.validateSNINames(obj)...)

	return errs
}

func (a apiserverV1) validateSNINames(obj *configv1.APIServer) field.ErrorList {
	errs := field.ErrorList{}
	if len(obj.Spec.ServingCerts.NamedCertificates) == 0 {
		return errs
	}

	infrastructure, err := a.infrastructureGetter().Infrastructures().Get(context.TODO(), "cluster", metav1.GetOptions{})
	if err != nil {
		errs = append(errs, field.InternalError(field.NewPath("metadata"), err))
	}
	for i, currSNI := range obj.Spec.ServingCerts.NamedCertificates {
		// if names are specified, confirm they do not match
		// if names are not specified, the cert can still match, but only the operator resolves the secrets down.  We gain a lot of benefit by being sure
		// we don't allow an explicit override of these values
		for j, currName := range currSNI.Names {
			path := field.NewPath("spec").Child("servingCerts").Index(i).Child("names").Index(j)
			if currName == infrastructure.Status.APIServerInternalURL {
				errs = append(errs, field.Invalid(path, currName, fmt.Sprintf("may not match internal loadbalancer: %q", infrastructure.Status.APIServerInternalURL)))
				continue
			}
			if strings.HasSuffix(currName, ".*") {
				withoutSuffix := currName[0 : len(currName)-2]
				if strings.HasPrefix(infrastructure.Status.APIServerInternalURL, withoutSuffix) {
					errs = append(errs, field.Invalid(path, currName, fmt.Sprintf("may not match internal loadbalancer: %q", infrastructure.Status.APIServerInternalURL)))
				}
			}
		}
	}

	return errs
}

func (a apiserverV1) ValidateUpdate(_ context.Context, uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
	obj, errs := toAPIServerV1(uncastObj)
	if len(errs) > 0 {
		return errs
	}
	oldObj, errs := toAPIServerV1(uncastOldObj)
	if len(errs) > 0 {
		return errs
	}

	errs = append(errs, validation.ValidateObjectMetaUpdate(&obj.ObjectMeta, &oldObj.ObjectMeta, field.NewPath("metadata"))...)
	errs = append(errs, validateAPIServerSpecUpdate(obj.Spec, oldObj.Spec)...)
	errs = append(errs, a.validateSNINames(obj)...)

	return errs
}

func (apiserverV1) ValidateStatusUpdate(_ context.Context, uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
	obj, errs := toAPIServerV1(uncastObj)
	if len(errs) > 0 {
		return errs
	}
	oldObj, errs := toAPIServerV1(uncastOldObj)
	if len(errs) > 0 {
		return errs
	}

	// TODO validate the obj.  remember that status validation should *never* fail on spec validation errors.
	errs = append(errs, validation.ValidateObjectMetaUpdate(&obj.ObjectMeta, &oldObj.ObjectMeta, field.NewPath("metadata"))...)
	errs = append(errs, validateAPIServerStatus(obj.Status)...)

	return errs
}

func validateAPIServerSpecCreate(spec configv1.APIServerSpec) field.ErrorList {
	errs := field.ErrorList{}
	specPath := field.NewPath("spec")

	errs = append(errs, validateAdditionalCORSAllowedOrigins(specPath.Child("additionalCORSAllowedOrigins"), spec.AdditionalCORSAllowedOrigins)...)
	errs = append(errs, validateTLSSecurityProfile(specPath.Child("tlsSecurityProfile"), spec.TLSSecurityProfile)...)

	return errs
}

func validateAPIServerSpecUpdate(newSpec, oldSpec configv1.APIServerSpec) field.ErrorList {
	errs := field.ErrorList{}
	specPath := field.NewPath("spec")

	errs = append(errs, validateAdditionalCORSAllowedOrigins(specPath.Child("additionalCORSAllowedOrigins"), newSpec.AdditionalCORSAllowedOrigins)...)
	errs = append(errs, validateTLSSecurityProfile(specPath.Child("tlsSecurityProfile"), newSpec.TLSSecurityProfile)...)

	return errs
}

func validateAPIServerStatus(status configv1.APIServerStatus) field.ErrorList {
	errs := field.ErrorList{}

	// TODO

	return errs
}

func validateAdditionalCORSAllowedOrigins(fieldPath *field.Path, cors []string) field.ErrorList {
	errs := field.ErrorList{}

	for i, re := range cors {
		if _, err := regexp.Compile(re); err != nil {
			errs = append(errs, field.Invalid(fieldPath.Index(i), re, fmt.Sprintf("not a valid regular expression: %v", err)))
		}
	}

	return errs
}

func validateTLSSecurityProfile(fieldPath *field.Path, profile *configv1.TLSSecurityProfile) field.ErrorList {
	errs := field.ErrorList{}

	if profile == nil {
		return errs
	}

	errs = append(errs, validateTLSSecurityProfileType(fieldPath, profile)...)

	if profile.Type == configv1.TLSProfileCustomType && profile.Custom != nil {
		errs = append(errs, validateCipherSuites(fieldPath.Child("custom", "ciphers"), profile.Custom.Ciphers, profile.Custom.MinTLSVersion)...)
		errs = append(errs, validateMinTLSVersion(fieldPath.Child("custom", "minTLSVersion"), profile.Custom.MinTLSVersion)...)
	}

	return errs
}

func validateTLSSecurityProfileType(fieldPath *field.Path, profile *configv1.TLSSecurityProfile) field.ErrorList {
	const typeProfileMismatchFmt = "type set to %s, but the corresponding field is unset"
	typePath := fieldPath.Child("type")

	errs := field.ErrorList{}

	switch profile.Type {
	case "":
		if profile.Old != nil || profile.Intermediate != nil || profile.Modern != nil || profile.Custom != nil {
			errs = append(errs, field.Required(typePath, "one of the profiles is set but 'type' field is empty"))
		}
	case configv1.TLSProfileOldType:
		if profile.Old == nil {
			errs = append(errs, field.Required(fieldPath.Child("old"), fmt.Sprintf(typeProfileMismatchFmt, profile.Type)))
		}
	case configv1.TLSProfileIntermediateType:
		if profile.Intermediate == nil {
			errs = append(errs, field.Required(fieldPath.Child("intermediate"), fmt.Sprintf(typeProfileMismatchFmt, profile.Type)))
		}
	case configv1.TLSProfileModernType:
		if profile.Modern == nil {
			errs = append(errs, field.Required(fieldPath.Child("modern"), fmt.Sprintf(typeProfileMismatchFmt, profile.Type)))
		}
	case configv1.TLSProfileCustomType:
		if profile.Custom == nil {
			errs = append(errs, field.Required(fieldPath.Child("custom"), fmt.Sprintf(typeProfileMismatchFmt, profile.Type)))
		}
	default:
		errs = append(errs, field.Invalid(typePath, profile.Type, fmt.Sprintf("unknown type, valid values are: [Old Intermediate Modern Custom]")))
	}

	return errs
}

func validateCipherSuites(fieldPath *field.Path, suites []string, version configv1.TLSProtocolVersion) field.ErrorList {
	errs := field.ErrorList{}

	if version == configv1.VersionTLS13 {
		if len(suites) != 0 {
			errs = append(errs, field.Invalid(fieldPath, suites, "TLS 1.3 cipher suites are not configurable"))
		}
		return errs
	}

	if ianaSuites := libgocrypto.OpenSSLToIANACipherSuites(suites); len(ianaSuites) == 0 {
		errs = append(errs, field.Invalid(fieldPath, suites, "no supported cipher suite found"))
	}

	// Return an error if it is missing ECDHE_RSA_WITH_AES_128_GCM_SHA256 or
	// ECDHE_ECDSA_WITH_AES_128_GCM_SHA256 to prevent the http2 Server
	// configuration to return an error when http2 required cipher suites aren't
	// provided.
	// See: go/x/net/http2.ConfigureServer for futher information.
	if !haveRequiredHTTP2CipherSuites(suites) {
		errs = append(errs, field.Invalid(fieldPath, suites, "http2: TLSConfig.CipherSuites is missing an HTTP/2-required AES_128_GCM_SHA256 cipher (need at least one of ECDHE-RSA-AES128-GCM-SHA256 or ECDHE-ECDSA-AES128-GCM-SHA256)"))
	}

	return errs
}

func haveRequiredHTTP2CipherSuites(suites []string) bool {
	for _, s := range suites {
		switch s {
		case "ECDHE-RSA-AES128-GCM-SHA256",
			// Alternative MTI cipher to not discourage ECDSA-only servers.
			// See http://golang.org/cl/30721 for further information.
			"ECDHE-ECDSA-AES128-GCM-SHA256":
			return true
		}
	}
	return false
}

func validateMinTLSVersion(fieldPath *field.Path, version configv1.TLSProtocolVersion) field.ErrorList {
	errs := field.ErrorList{}
	if _, err := libgocrypto.TLSVersion(string(version)); err != nil {
		errs = append(errs, field.Invalid(fieldPath, version, err.Error()))
	}
	return errs
}
