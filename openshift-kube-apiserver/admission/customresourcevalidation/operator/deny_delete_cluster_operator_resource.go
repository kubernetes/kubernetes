package operator

import (
	"context"
	"fmt"
	"io"

	"k8s.io/apiserver/pkg/admission"
)

const PluginName = "operator.openshift.io/DenyDeleteClusterOperators"

// Register registers an admission plugin factory whose plugin prevents the deletion of cluster operator resources.
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return newAdmissionPlugin(), nil
	})
}

var _ admission.ValidationInterface = &admissionPlugin{}

type admissionPlugin struct {
	*admission.Handler
}

func newAdmissionPlugin() *admissionPlugin {
	return &admissionPlugin{Handler: admission.NewHandler(admission.Delete)}
}

// Validate returns an error if there is an attempt to delete a cluster operator resource.
func (p *admissionPlugin) Validate(ctx context.Context, attributes admission.Attributes, _ admission.ObjectInterfaces) error {
	if len(attributes.GetSubresource()) > 0 {
		return nil
	}
	if attributes.GetResource().Group != "operator.openshift.io" {
		return nil
	}
	switch attributes.GetResource().Resource {
	// Deletion is denied for storages.operator.openshift.io objects named cluster,
	// because MCO and KCM-O depend on this resource being present in order to
	// correctly set environment variables on kubelet and kube-controller-manager.
	case "storages":
		if attributes.GetName() != "cluster" {
			return nil
		}
	// Deletion is allowed for all other operator.openshift.io objects unless
	// explicitly listed above.
	default:
		return nil
	}
	return admission.NewForbidden(attributes, fmt.Errorf("deleting required %s.%s resource, named %s, is not allowed", attributes.GetResource().Resource, attributes.GetResource().Group, attributes.GetName()))
}
