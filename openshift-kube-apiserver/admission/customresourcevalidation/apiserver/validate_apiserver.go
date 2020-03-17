package apiserver

import (
	"fmt"
	"regexp"
	"strings"

	configv1 "github.com/openshift/api/config/v1"
	configv1client "github.com/openshift/client-go/config/clientset/versioned/typed/config/v1"

	"k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation"
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

	errs := validateAdditionalCORSAllowedOrigins(field.NewPath("spec").Child("additionalCORSAllowedOrigins"), spec.AdditionalCORSAllowedOrigins)
	return errs
}

func validateAPIServerSpecUpdate(newSpec, oldSpec configv1.APIServerSpec) field.ErrorList {

	errs := validateAdditionalCORSAllowedOrigins(field.NewPath("spec").Child("additionalCORSAllowedOrigins"), newSpec.AdditionalCORSAllowedOrigins)
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
