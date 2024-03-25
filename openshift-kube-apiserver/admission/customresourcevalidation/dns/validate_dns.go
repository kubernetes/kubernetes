package dns

import (
	"context"
	"fmt"
	"io"
	"reflect"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/validation"
	unversionedvalidation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
	api "k8s.io/kubernetes/pkg/apis/core"
	k8s_api_v1 "k8s.io/kubernetes/pkg/apis/core/v1"
	apivalidation "k8s.io/kubernetes/pkg/apis/core/validation"

	operatorv1 "github.com/openshift/api/operator/v1"
	crvalidation "k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation"
)

const PluginName = "operator.openshift.io/ValidateDNS"

// Register registers the DNS validation plugin.
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return crvalidation.NewValidator(
			map[schema.GroupResource]bool{
				operatorv1.GroupVersion.WithResource("dnses").GroupResource(): true,
			},
			map[schema.GroupVersionKind]crvalidation.ObjectValidator{
				operatorv1.GroupVersion.WithKind("DNS"): dnsV1{},
			})
	})
}

// toDNSV1 converts a runtime object to a versioned DNS.
func toDNSV1(uncastObj runtime.Object) (*operatorv1.DNS, field.ErrorList) {
	if uncastObj == nil {
		return nil, nil
	}

	obj, ok := uncastObj.(*operatorv1.DNS)
	if !ok {
		return nil, field.ErrorList{
			field.NotSupported(field.NewPath("kind"), fmt.Sprintf("%T", uncastObj), []string{"DNS"}),
			field.NotSupported(field.NewPath("apiVersion"), fmt.Sprintf("%T", uncastObj), []string{"operator.openshift.io/v1"}),
		}
	}

	return obj, nil
}

// dnsV1 is runtime object that is validated as a versioned DNS.
type dnsV1 struct{}

// ValidateCreate validates a DNS that is being created.
func (dnsV1) ValidateCreate(_ context.Context, uncastObj runtime.Object) field.ErrorList {
	obj, errs := toDNSV1(uncastObj)
	if len(errs) > 0 {
		return errs
	}

	errs = append(errs, validation.ValidateObjectMeta(&obj.ObjectMeta, false, validation.NameIsDNSSubdomain, field.NewPath("metadata"))...)
	errs = append(errs, validateDNSSpecCreate(obj.Spec)...)

	return errs
}

// ValidateUpdate validates a DNS that is being updated.
func (dnsV1) ValidateUpdate(_ context.Context, uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
	obj, errs := toDNSV1(uncastObj)
	if len(errs) > 0 {
		return errs
	}
	oldObj, errs := toDNSV1(uncastOldObj)
	if len(errs) > 0 {
		return errs
	}

	errs = append(errs, validation.ValidateObjectMetaUpdate(&obj.ObjectMeta, &oldObj.ObjectMeta, field.NewPath("metadata"))...)
	errs = append(errs, validateDNSSpecUpdate(obj.Spec, oldObj.Spec)...)

	return errs
}

// ValidateStatusUpdate validates a DNS status that is being updated.
func (dnsV1) ValidateStatusUpdate(_ context.Context, uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
	obj, errs := toDNSV1(uncastObj)
	if len(errs) > 0 {
		return errs
	}
	oldObj, errs := toDNSV1(uncastOldObj)
	if len(errs) > 0 {
		return errs
	}

	errs = append(errs, validation.ValidateObjectMetaUpdate(&obj.ObjectMeta, &oldObj.ObjectMeta, field.NewPath("metadata"))...)

	return errs
}

// validateDNSSpecCreate validates the spec of a DNS that is being created.
func validateDNSSpecCreate(spec operatorv1.DNSSpec) field.ErrorList {
	var errs field.ErrorList
	specField := field.NewPath("spec")
	errs = append(errs, validateDNSNodePlacement(spec.NodePlacement, specField.Child("nodePlacement"))...)
	errs = append(errs, validateUpstreamResolversCreate(spec.UpstreamResolvers, specField.Child("upstreamResolvers"))...)
	errs = append(errs, validateServersCreate(spec.Servers, specField.Child("servers"))...)
	return errs
}

// validateDNSSpecUpdate validates the spec of a DNS that is being updated.
func validateDNSSpecUpdate(newspec, oldspec operatorv1.DNSSpec) field.ErrorList {
	var errs field.ErrorList
	specField := field.NewPath("spec")
	errs = append(errs, validateDNSNodePlacement(newspec.NodePlacement, specField.Child("nodePlacement"))...)
	errs = append(errs, validateUpstreamResolversUpdate(newspec.UpstreamResolvers, oldspec.UpstreamResolvers, specField.Child("upstreamResolvers"))...)
	errs = append(errs, validateServersUpdate(newspec.Servers, oldspec.Servers, specField.Child("servers"))...)
	return errs
}

// validateDNSSpec validates the spec.nodePlacement field of a DNS.
func validateDNSNodePlacement(nodePlacement operatorv1.DNSNodePlacement, fldPath *field.Path) field.ErrorList {
	var errs field.ErrorList
	if len(nodePlacement.NodeSelector) != 0 {
		errs = append(errs, unversionedvalidation.ValidateLabels(nodePlacement.NodeSelector, fldPath.Child("nodeSelector"))...)
	}
	if len(nodePlacement.Tolerations) != 0 {
		errs = append(errs, validateTolerations(nodePlacement.Tolerations, fldPath.Child("tolerations"))...)
	}
	return errs
}

// validateTolerations validates a slice of corev1.Toleration.
func validateTolerations(versionedTolerations []corev1.Toleration, fldPath *field.Path) field.ErrorList {
	allErrors := field.ErrorList{}
	unversionedTolerations := make([]api.Toleration, len(versionedTolerations))
	for i := range versionedTolerations {
		if err := k8s_api_v1.Convert_v1_Toleration_To_core_Toleration(&versionedTolerations[i], &unversionedTolerations[i], nil); err != nil {
			allErrors = append(allErrors, field.Invalid(fldPath.Index(i), unversionedTolerations[i], err.Error()))
		}
	}
	allErrors = append(allErrors, apivalidation.ValidateTolerations(unversionedTolerations, fldPath)...)
	return allErrors
}

// validateUpstreamResolversCreate validates configuration of the Upstream objects when TLS is configured.
func validateUpstreamResolversCreate(upstreamResolvers operatorv1.UpstreamResolvers, fieldPath *field.Path) field.ErrorList {
	var errs field.ErrorList

	errs = append(errs, validateDNSTransportConfig(upstreamResolvers.TransportConfig, fieldPath.Child("transportConfig"))...)

	if upstreamResolvers.TransportConfig.Transport == operatorv1.TLSTransport {
		// Transport is TLS so we must check if there are mixed Upstream types. SystemResolveConf is not allowed with TLS.
		for i, upstream := range upstreamResolvers.Upstreams {
			if upstream.Type == operatorv1.SystemResolveConfType {
				errMessage := "SystemResolvConf is not allowed when TLS is configured as the transport"
				errs = append(errs, field.Invalid(fieldPath.Child("upstreams").Index(i).Child("type"), upstream.Type, errMessage))
			}
		}
	}

	return errs
}

// validateUpstreamResolversUpdate validates configuration of the Upstream objects when TLS is configured.
func validateUpstreamResolversUpdate(newUpstreamResolvers operatorv1.UpstreamResolvers, oldUpstreamResolvers operatorv1.UpstreamResolvers, fieldPath *field.Path) field.ErrorList {
	var errs field.ErrorList
	newTransport := newUpstreamResolvers.TransportConfig.Transport

	if !reflect.DeepEqual(newUpstreamResolvers.TransportConfig, oldUpstreamResolvers.TransportConfig) || isKnownTransport(newTransport) {
		errs = append(errs, validateUpstreamResolversCreate(newUpstreamResolvers, fieldPath)...)
	}

	return errs
}

func isKnownTransport(transport operatorv1.DNSTransport) bool {
	switch transport {
	case "", operatorv1.CleartextTransport, operatorv1.TLSTransport:
		return true
	default:
		return false
	}

}

func validateServersCreate(servers []operatorv1.Server, fieldPath *field.Path) field.ErrorList {
	var errs field.ErrorList
	for i, server := range servers {
		errs = append(errs, validateDNSTransportConfig(server.ForwardPlugin.TransportConfig, fieldPath.Index(i).Child("forwardPlugin").Child("transportConfig"))...)
	}
	return errs
}

func validateServersUpdate(newServers []operatorv1.Server, oldServers []operatorv1.Server, fieldPath *field.Path) field.ErrorList {
	var errs field.ErrorList
	for i, newServer := range newServers {
		for _, oldServer := range oldServers {
			// Use server.Name as the pivot for comparison since a cluster admin could conceivably change the transport
			// and/or upstreams, making those insufficient for comparison.
			if newServer.Name == oldServer.Name {
				// TransportConfig has changed
				if !reflect.DeepEqual(newServer.ForwardPlugin.TransportConfig, oldServer.ForwardPlugin.TransportConfig) {
					errs = append(validateDNSTransportConfig(newServer.ForwardPlugin.TransportConfig, fieldPath.Index(i).Child("forwardPlugin").Child("transportConfig")))
				}
			}
		}
	}
	return errs
}

func validateDNSTransportConfig(transportConfig operatorv1.DNSTransportConfig, fieldPath *field.Path) field.ErrorList {
	var errs field.ErrorList
	var emptyTransportConfig operatorv1.DNSTransportConfig
	tlsConfig := transportConfig.TLS

	// No validation is needed on an empty TransportConfig.
	if transportConfig == emptyTransportConfig {
		return errs
	}

	switch transportConfig.Transport {
	case "", operatorv1.CleartextTransport:
		// Don't allow TLS configuration when using empty or Cleartext
		if tlsConfig != nil {
			errs = append(errs, field.Invalid(fieldPath.Child("tls"), transportConfig.TLS, "TLS must not be configured when using an empty or cleartext transport"))
		}
	case operatorv1.TLSTransport:
		// When Transport is TLS, there MUST be a ServerName configured.
		if tlsConfig == nil || tlsConfig.ServerName == "" {
			errs = append(errs, field.Required(fieldPath.Child("tls").Child("serverName"), "transportConfig requires a serverName when transport is TLS"))
		}
	default:
		errs = append(errs, field.Invalid(fieldPath.Child("transport"), transportConfig.Transport, "unknown transport"))
	}

	return errs
}
