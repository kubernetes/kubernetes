package config

import (
	"context"
	"fmt"
	"io"

	"k8s.io/apiserver/pkg/admission"
)

const PluginName = "config.openshift.io/DenyDeleteClusterConfiguration"

// Register registers an admission plugin factory whose plugin prevents the deletion of cluster configuration resources.
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

// Validate returns an error if there is an attempt to delete a cluster configuration resource.
func (p *admissionPlugin) Validate(ctx context.Context, attributes admission.Attributes, _ admission.ObjectInterfaces) error {
	if len(attributes.GetSubresource()) > 0 {
		return nil
	}
	if attributes.GetResource().Group != "config.openshift.io" {
		return nil
	}
	// clusteroperators can be deleted so that we can force status refreshes and change over time.
	// clusterversions not named `version` can be deleted (none are expected to exist).
	// other config.openshift.io resources not named `cluster` can be deleted (none are expected to exist).
	switch attributes.GetResource().Resource {
	case "clusteroperators":
		return nil
	case "clusterversions":
		if attributes.GetName() != "version" {
			return nil
		}
	default:
		if attributes.GetName() != "cluster" {
			return nil
		}
	}
	return admission.NewForbidden(attributes, fmt.Errorf("deleting required %s.%s resource, named %s, is not allowed", attributes.GetResource().Resource, attributes.GetResource().Group, attributes.GetName()))
}
