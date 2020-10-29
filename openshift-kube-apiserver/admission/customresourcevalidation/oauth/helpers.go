package oauth

import (
	"net"

	kvalidation "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"

	configv1 "github.com/openshift/api/config/v1"
	"github.com/openshift/library-go/pkg/config/validation"
	crvalidation "k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation"
)

func isValidHostname(hostname string) bool {
	return len(kvalidation.IsDNS1123Subdomain(hostname)) == 0 || net.ParseIP(hostname) != nil
}

func ValidateRemoteConnectionInfo(remoteConnectionInfo configv1.OAuthRemoteConnectionInfo, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(remoteConnectionInfo.URL) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("url"), ""))
	} else {
		_, urlErrs := validation.ValidateSecureURL(remoteConnectionInfo.URL, fldPath.Child("url"))
		allErrs = append(allErrs, urlErrs...)
	}

	allErrs = append(allErrs, crvalidation.ValidateConfigMapReference(fldPath.Child("ca"), remoteConnectionInfo.CA, false)...)
	allErrs = append(allErrs, crvalidation.ValidateSecretReference(fldPath.Child("tlsClientCert"), remoteConnectionInfo.TLSClientCert, false)...)
	allErrs = append(allErrs, crvalidation.ValidateSecretReference(fldPath.Child("tlsClientKey"), remoteConnectionInfo.TLSClientKey, false)...)

	return allErrs
}
