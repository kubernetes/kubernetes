package csiinlinevolumesecurity

import (
	"context"
	"fmt"
	"io"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/warning"
	"k8s.io/client-go/informers"
	corev1listers "k8s.io/client-go/listers/core/v1"
	storagev1listers "k8s.io/client-go/listers/storage/v1"
	"k8s.io/klog/v2"
	appsapi "k8s.io/kubernetes/pkg/apis/apps"
	batchapi "k8s.io/kubernetes/pkg/apis/batch"
	coreapi "k8s.io/kubernetes/pkg/apis/core"
	podsecapi "k8s.io/pod-security-admission/api"
)

const (
	// Plugin name
	PluginName = "storage.openshift.io/CSIInlineVolumeSecurity"
	// Label on the CSIDriver to declare the driver's effective pod security profile
	csiInlineVolProfileLabel = "security.openshift.io/csi-ephemeral-volume-profile"
	// Default values for the profile labels when no such label exists
	defaultCSIInlineVolProfile  = podsecapi.LevelPrivileged
	defaultPodSecEnforceProfile = podsecapi.LevelRestricted
	defaultPodSecWarnProfile    = podsecapi.LevelRestricted
	defaultPodSecAuditProfile   = podsecapi.LevelRestricted
	// Format string used for audit/warn/enforce response messages
	admissionResponseFormatStr = "%s uses an inline volume provided by CSIDriver %s and namespace %s has a pod security %s level that is lower than %s"
)

var (
	podSpecResources = map[schema.GroupResource]bool{
		coreapi.Resource("pods"):                   true,
		coreapi.Resource("replicationcontrollers"): true,
		coreapi.Resource("podtemplates"):           true,
		appsapi.Resource("replicasets"):            true,
		appsapi.Resource("deployments"):            true,
		appsapi.Resource("statefulsets"):           true,
		appsapi.Resource("daemonsets"):             true,
		batchapi.Resource("jobs"):                  true,
		batchapi.Resource("cronjobs"):              true,
	}
)

var _ = initializer.WantsExternalKubeInformerFactory(&csiInlineVolSec{})
var _ = admission.ValidationInterface(&csiInlineVolSec{})

func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName,
		func(config io.Reader) (admission.Interface, error) {
			return &csiInlineVolSec{
				Handler: admission.NewHandler(admission.Create),
			}, nil
		})
}

// csiInlineVolSec validates whether the namespace has permission to use a given
// CSI driver as an inline volume.
type csiInlineVolSec struct {
	*admission.Handler
	//enabled               bool
	//inspectedFeatureGates bool
	defaultPolicy       podsecapi.Policy
	nsLister            corev1listers.NamespaceLister
	nsListerSynced      func() bool
	csiDriverLister     storagev1listers.CSIDriverLister
	csiDriverListSynced func() bool
	podSpecExtractor    PodSpecExtractor
}

// SetExternalKubeInformerFactory registers an informer
func (c *csiInlineVolSec) SetExternalKubeInformerFactory(kubeInformers informers.SharedInformerFactory) {
	c.nsLister = kubeInformers.Core().V1().Namespaces().Lister()
	c.nsListerSynced = kubeInformers.Core().V1().Namespaces().Informer().HasSynced
	c.csiDriverLister = kubeInformers.Storage().V1().CSIDrivers().Lister()
	c.csiDriverListSynced = kubeInformers.Storage().V1().CSIDrivers().Informer().HasSynced
	c.podSpecExtractor = &OCPPodSpecExtractor{}
	c.SetReadyFunc(func() bool {
		return c.nsListerSynced() && c.csiDriverListSynced()
	})

	// set default pod security policy
	c.defaultPolicy = podsecapi.Policy{
		Enforce: podsecapi.LevelVersion{
			Level:   defaultPodSecEnforceProfile,
			Version: podsecapi.GetAPIVersion(),
		},
		Warn: podsecapi.LevelVersion{
			Level:   defaultPodSecWarnProfile,
			Version: podsecapi.GetAPIVersion(),
		},
		Audit: podsecapi.LevelVersion{
			Level:   defaultPodSecAuditProfile,
			Version: podsecapi.GetAPIVersion(),
		},
	}
}

func (c *csiInlineVolSec) ValidateInitialization() error {
	if c.nsLister == nil {
		return fmt.Errorf("%s plugin needs a namespace lister", PluginName)
	}
	if c.nsListerSynced == nil {
		return fmt.Errorf("%s plugin needs a namespace lister synced", PluginName)
	}
	if c.csiDriverLister == nil {
		return fmt.Errorf("%s plugin needs a node lister", PluginName)
	}
	if c.csiDriverListSynced == nil {
		return fmt.Errorf("%s plugin needs a node lister synced", PluginName)
	}
	if c.podSpecExtractor == nil {
		return fmt.Errorf("%s plugin needs a pod spec extractor", PluginName)
	}
	return nil
}

func (c *csiInlineVolSec) PolicyToEvaluate(labels map[string]string) (podsecapi.Policy, field.ErrorList) {
	return podsecapi.PolicyToEvaluate(labels, c.defaultPolicy)
}

func (c *csiInlineVolSec) Validate(ctx context.Context, attrs admission.Attributes, o admission.ObjectInterfaces) error {
	// Only validate applicable resources
	gr := attrs.GetResource().GroupResource()
	if !podSpecResources[gr] {
		return nil
	}
	// Do not validate subresources
	if attrs.GetSubresource() != "" {
		return nil
	}

	// Get namespace
	namespace, err := c.nsLister.Get(attrs.GetNamespace())
	if err != nil {
		return admission.NewForbidden(attrs, fmt.Errorf("failed to get namespace: %v", err))
	}
	// Require valid labels if they exist (the default policy is always valid)
	nsPolicy, nsPolicyErrs := c.PolicyToEvaluate(namespace.Labels)
	if len(nsPolicyErrs) > 0 {
		return admission.NewForbidden(attrs, fmt.Errorf("invalid policy found on namespace %s: %v", namespace, nsPolicyErrs))
	}
	// If the namespace policy is fully privileged, no need to evaluate further
	// because it is allowed to use any inline volumes.
	if nsPolicy.FullyPrivileged() {
		return nil
	}

	// Extract the pod spec to evaluate
	obj := attrs.GetObject()
	_, podSpec, err := c.podSpecExtractor.ExtractPodSpec(obj)
	if err != nil {
		return admission.NewForbidden(attrs, fmt.Errorf("failed to extract pod spec: %v", err))
	}
	// If an object with an optional pod spec does not contain a pod spec, skip validation
	if podSpec == nil {
		return nil
	}

	klogV := klog.V(5)
	if klogV.Enabled() {
		klogV.InfoS("CSIInlineVolumeSecurity evaluation", "policy", fmt.Sprintf("%v", nsPolicy), "op", attrs.GetOperation(), "resource", attrs.GetResource(), "namespace", attrs.GetNamespace(), "name", attrs.GetName())
	}

	// For each inline volume, find the CSIDriver and ensure the profile on the
	// driver is allowed by the pod security profile on the namespace.
	// If it is not: create errors, warnings, and audit as defined by policy.
	for _, vol := range podSpec.Volumes {
		// Only check for inline volumes
		if vol.CSI == nil {
			continue
		}

		// Get the policy level for the CSIDriver
		driverName := vol.CSI.Driver
		driverLevel, err := c.getCSIDriverLevel(driverName)
		if err != nil {
			return admission.NewForbidden(attrs, err)
		}

		// Compare CSIDriver level to the policy for the namespace
		if podsecapi.CompareLevels(nsPolicy.Enforce.Level, driverLevel) > 0 {
			// Not permitted, enforce error and deny admission
			return admission.NewForbidden(attrs, fmt.Errorf(admissionResponseFormatStr, attrs.GetName(), driverName, attrs.GetNamespace(), "enforce", driverLevel))
		}
		if podsecapi.CompareLevels(nsPolicy.Warn.Level, driverLevel) > 0 {
			// Violates policy warn level, add warning
			warning.AddWarning(ctx, "", fmt.Sprintf(admissionResponseFormatStr, attrs.GetName(), driverName, attrs.GetNamespace(), "warn", driverLevel))
		}
		if podsecapi.CompareLevels(nsPolicy.Audit.Level, driverLevel) > 0 {
			// Violates policy audit level, add audit annotation
			auditMessageString := fmt.Sprintf(admissionResponseFormatStr, attrs.GetName(), driverName, attrs.GetNamespace(), "audit", driverLevel)
			audit.AddAuditAnnotation(ctx, PluginName, auditMessageString)
		}
	}

	return nil
}

// getCSIDriverLevel returns the effective policy level for the CSIDriver.
// If the driver is found and it has the label, use that policy.
// If the driver or the label is missing, default to the privileged policy.
func (c *csiInlineVolSec) getCSIDriverLevel(driverName string) (podsecapi.Level, error) {
	driverLevel := defaultCSIInlineVolProfile
	driver, err := c.csiDriverLister.Get(driverName)
	if err != nil {
		return driverLevel, nil
	}

	csiDriverLabel, ok := driver.ObjectMeta.Labels[csiInlineVolProfileLabel]
	if !ok {
		return driverLevel, nil
	}

	driverLevel, err = podsecapi.ParseLevel(csiDriverLabel)
	if err != nil {
		return driverLevel, fmt.Errorf("invalid label %s for CSIDriver %s: %v", csiInlineVolProfileLabel, driverName, err)
	}

	return driverLevel, nil
}

// PodSpecExtractor extracts a PodSpec from pod-controller resources that embed a PodSpec.
// This is the same as what is used in the pod-security-admission plugin (see
// staging/src/k8s.io/pod-security-admission/admission/admission.go) except here we
// are provided coreapi resources instead of corev1, which changes the interface.
type PodSpecExtractor interface {
	// HasPodSpec returns true if the given resource type MAY contain an extractable PodSpec.
	HasPodSpec(schema.GroupResource) bool
	// ExtractPodSpec returns a pod spec and metadata to evaluate from the object.
	// An error returned here does not block admission of the pod-spec-containing object and is not returned to the user.
	// If the object has no pod spec, return `nil, nil, nil`.
	ExtractPodSpec(runtime.Object) (*metav1.ObjectMeta, *coreapi.PodSpec, error)
}

type OCPPodSpecExtractor struct{}

func (OCPPodSpecExtractor) HasPodSpec(gr schema.GroupResource) bool {
	return podSpecResources[gr]
}

func (OCPPodSpecExtractor) ExtractPodSpec(obj runtime.Object) (*metav1.ObjectMeta, *coreapi.PodSpec, error) {
	switch o := obj.(type) {
	case *coreapi.Pod:
		return &o.ObjectMeta, &o.Spec, nil
	case *coreapi.PodTemplate:
		return extractPodSpecFromTemplate(&o.Template)
	case *coreapi.ReplicationController:
		return extractPodSpecFromTemplate(o.Spec.Template)
	case *appsapi.ReplicaSet:
		return extractPodSpecFromTemplate(&o.Spec.Template)
	case *appsapi.Deployment:
		return extractPodSpecFromTemplate(&o.Spec.Template)
	case *appsapi.DaemonSet:
		return extractPodSpecFromTemplate(&o.Spec.Template)
	case *appsapi.StatefulSet:
		return extractPodSpecFromTemplate(&o.Spec.Template)
	case *batchapi.Job:
		return extractPodSpecFromTemplate(&o.Spec.Template)
	case *batchapi.CronJob:
		return extractPodSpecFromTemplate(&o.Spec.JobTemplate.Spec.Template)
	default:
		return nil, nil, fmt.Errorf("unexpected object type: %s", obj.GetObjectKind().GroupVersionKind().String())
	}
}

func extractPodSpecFromTemplate(template *coreapi.PodTemplateSpec) (*metav1.ObjectMeta, *coreapi.PodSpec, error) {
	if template == nil {
		return nil, nil, nil
	}
	return &template.ObjectMeta, &template.Spec, nil
}
