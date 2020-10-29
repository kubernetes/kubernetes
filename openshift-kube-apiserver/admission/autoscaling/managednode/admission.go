package managednode

import (
	"context"
	"fmt"
	"io"
	"strings"

	configv1 "github.com/openshift/api/config/v1"
	configv1informer "github.com/openshift/client-go/config/informers/externalversions/config/v1"
	configv1listers "github.com/openshift/client-go/config/listers/config/v1"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/kubernetes/pkg/kubelet/managed"

	corev1 "k8s.io/api/core/v1"
	coreapi "k8s.io/kubernetes/pkg/apis/core"

	"k8s.io/client-go/kubernetes"
)

const (
	PluginName = "autoscaling.openshift.io/ManagedNode"
	// infraClusterName contains the name of the cluster infrastructure resource
	infraClusterName = "cluster"
)

var _ = initializer.WantsExternalKubeClientSet(&managedNodeValidate{})
var _ = admission.ValidationInterface(&managedNodeValidate{})
var _ = WantsInfraInformer(&managedNodeValidate{})

func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName,
		func(_ io.Reader) (admission.Interface, error) {
			return &managedNodeValidate{
				Handler: admission.NewHandler(admission.Create, admission.Update),
			}, nil
		})
}

type managedNodeValidate struct {
	*admission.Handler
	client                kubernetes.Interface
	infraConfigLister     configv1listers.InfrastructureLister
	infraConfigListSynced func() bool
}

// SetExternalKubeClientSet implements the WantsExternalKubeClientSet interface.
func (a *managedNodeValidate) SetExternalKubeClientSet(client kubernetes.Interface) {
	a.client = client
}

func (a *managedNodeValidate) SetInfraInformer(informer configv1informer.InfrastructureInformer) {
	a.infraConfigLister = informer.Lister()
	a.infraConfigListSynced = informer.Informer().HasSynced
}

func (a *managedNodeValidate) ValidateInitialization() error {
	if a.client == nil {
		return fmt.Errorf("%s plugin needs a kubernetes client", PluginName)
	}
	if a.infraConfigLister == nil {
		return fmt.Errorf("%s did not get a config infrastructure lister", PluginName)
	}
	if a.infraConfigListSynced == nil {
		return fmt.Errorf("%s plugin needs a config infrastructure lister synced", PluginName)
	}
	return nil
}

func (a *managedNodeValidate) Validate(ctx context.Context, attr admission.Attributes, o admission.ObjectInterfaces) (err error) {
	if attr.GetResource().GroupResource() != corev1.Resource("nodes") || attr.GetSubresource() != "" {
		return nil
	}

	node, ok := attr.GetObject().(*coreapi.Node)
	if !ok {
		return admission.NewForbidden(attr, fmt.Errorf("unexpected object: %#v", attr.GetResource()))
	}

	// infraConfigListSynced is expected to be thread-safe since the underlying call is to the standard
	// informer HasSynced() function which is thread-safe.
	if !a.infraConfigListSynced() {
		return admission.NewForbidden(attr, fmt.Errorf("%s infra config cache not synchronized", PluginName))
	}

	clusterInfra, err := a.infraConfigLister.Get(infraClusterName)
	if err != nil {
		return admission.NewForbidden(attr, err) // can happen due to informer latency
	}

	// Check if we are in CPU Partitioning mode for AllNodes
	allErrs := validateClusterCPUPartitioning(clusterInfra.Status, node)
	if len(allErrs) == 0 {
		return nil
	}
	return errors.NewInvalid(attr.GetKind().GroupKind(), node.Name, allErrs)
}

// validateClusterCPUPartitioning Make sure that we only check nodes when CPU Partitioning is turned on.
// We also need to account for Single Node upgrades, during that initial upgrade, NTO will update this field during
// upgrade to make it authoritative from that point on. A roll back will revert an SingleNode cluster back to it's normal cycle.
// Other installations will have this field set at install time, and can not be turned off.
//
// If CPUPartitioning == AllNodes and is not empty value, check nodes
func validateClusterCPUPartitioning(infraStatus configv1.InfrastructureStatus, node *coreapi.Node) field.ErrorList {
	errorMessage := "node does not contain resource information, this is required for clusters with workload partitioning enabled"
	var allErrs field.ErrorList

	if infraStatus.CPUPartitioning == configv1.CPUPartitioningAllNodes {
		if !containsCPUResource(node.Status.Capacity) {
			allErrs = append(allErrs, getNodeInvalidWorkloadResourceError("capacity", errorMessage))
		}
		if !containsCPUResource(node.Status.Allocatable) {
			allErrs = append(allErrs, getNodeInvalidWorkloadResourceError("allocatable", errorMessage))
		}
	}

	return allErrs
}

func containsCPUResource(resources coreapi.ResourceList) bool {
	for k := range resources {
		if strings.Contains(k.String(), managed.WorkloadsCapacitySuffix) {
			return true
		}
	}
	return false
}

func getNodeInvalidWorkloadResourceError(resourcePool, message string) *field.Error {
	return field.Required(field.NewPath("status", resourcePool, managed.WorkloadsCapacitySuffix), message)
}
