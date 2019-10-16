package apiserver

import (
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

func (a apiserverV1) ValidateCreate(uncastObj runtime.Object) field.ErrorList {
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

	infrastructure, err := a.infrastructureGetter().Infrastructures().Get("cluster", metav1.GetOptions{})
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

func (a apiserverV1) ValidateUpdate(uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
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

func (apiserverV1) ValidateStatusUpdate(uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
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
		errs = append(errs, validateCipherSuites(fieldPath.Child("custom", "ciphers"), profile.Custom.Ciphers)...)
		errs = append(errs, validateMinTLSVersion(fieldPath.Child("custom", "minTLSVersion"), profile.Custom.MinTLSVersion)...)
	}

	return errs
}

func validateTLSSecurityProfileType(fieldPath *field.Path, profile *configv1.TLSSecurityProfile) field.ErrorList {
	const typeProfileMismatchFmt = "type set to %s, but the corresponding field is unset"
	typePath := fieldPath.Child("type")

	errs := field.ErrorList{}

	availableTypes := []string{
		string(configv1.TLSProfileOldType),
		string(configv1.TLSProfileIntermediateType),
		string(configv1.TLSProfileCustomType),
	}

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
		errs = append(errs, field.NotSupported(fieldPath.Child("type"), profile.Type, availableTypes))
	case configv1.TLSProfileCustomType:
		if profile.Custom == nil {
			errs = append(errs, field.Required(fieldPath.Child("custom"), fmt.Sprintf(typeProfileMismatchFmt, profile.Type)))
		}
	default:
		errs = append(errs, field.Invalid(typePath, profile.Type, fmt.Sprintf("unknown type, valid values are: %v", availableTypes)))
	}

	return errs
}

func validateCipherSuites(fieldPath *field.Path, suites []string) field.ErrorList {
	errs := field.ErrorList{}

	if ianaSuites := libgocrypto.OpenSSLToIANACipherSuites(suites); len(ianaSuites) == 0 {
		errs = append(errs, field.Invalid(fieldPath, suites, "no supported cipher suite found"))
	}

	return errs
}

func validateMinTLSVersion(fieldPath *field.Path, version configv1.TLSProtocolVersion) field.ErrorList {
	errs := field.ErrorList{}

	if version == configv1.VersionTLS13 {
		return append(errs, field.NotSupported(fieldPath, version, []string{string(configv1.VersionTLS10), string(configv1.VersionTLS11), string(configv1.VersionTLS12)}))
	}

	if _, err := libgocrypto.TLSVersion(string(version)); err != nil {
		errs = append(errs, field.Invalid(fieldPath, version, err.Error()))
	}

	return errs
}
