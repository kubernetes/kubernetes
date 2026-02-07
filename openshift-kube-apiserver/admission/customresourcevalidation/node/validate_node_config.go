package node

import (
	"context"
	"errors"
	"fmt"
	"io"

	configv1 "github.com/openshift/api/config/v1"
	nodelib "github.com/openshift/library-go/pkg/apiserver/node"

	openshiftfeatures "github.com/openshift/api/features"
	"k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/component-base/featuregate"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation"
)

var rejectionScenarios = []struct {
	fromProfile configv1.WorkerLatencyProfileType
	toProfile   configv1.WorkerLatencyProfileType
}{
	{fromProfile: "", toProfile: configv1.LowUpdateSlowReaction},
	{fromProfile: configv1.LowUpdateSlowReaction, toProfile: ""},
	{fromProfile: configv1.DefaultUpdateDefaultReaction, toProfile: configv1.LowUpdateSlowReaction},
	{fromProfile: configv1.LowUpdateSlowReaction, toProfile: configv1.DefaultUpdateDefaultReaction},
}

const PluginName = "config.openshift.io/ValidateConfigNodeV1"

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		ret := &configNodeV1Wrapper{}
		delegate, err := customresourcevalidation.NewValidator(
			map[schema.GroupResource]bool{
				configv1.Resource("nodes"): true,
			},
			map[schema.GroupVersionKind]customresourcevalidation.ObjectValidator{
				configv1.GroupVersion.WithKind("Node"): &configNodeV1{
					nodeListerFn:                 ret.getNodeLister,
					waitForNodeInformerSyncedFn:  ret.waitForNodeInformerSyncedFn,
					minimumKubeletVersionEnabled: feature.DefaultFeatureGate.Enabled(featuregate.Feature(openshiftfeatures.FeatureGateMinimumKubeletVersion)),
				},
			})
		if err != nil {
			return nil, err
		}
		ret.delegate = delegate
		return ret, nil
	})
}

func toConfigNodeV1(uncastObj runtime.Object) (*configv1.Node, field.ErrorList) {
	if uncastObj == nil {
		return nil, nil
	}

	allErrs := field.ErrorList{}

	obj, ok := uncastObj.(*configv1.Node)
	if !ok {
		return nil, append(allErrs,
			field.NotSupported(field.NewPath("kind"), fmt.Sprintf("%T", uncastObj), []string{"Node"}),
			field.NotSupported(field.NewPath("apiVersion"), fmt.Sprintf("%T", uncastObj), []string{"config.openshift.io/v1"}))
	}

	return obj, nil
}

type configNodeV1 struct {
	nodeListerFn                 func() corev1listers.NodeLister
	waitForNodeInformerSyncedFn  func() bool
	minimumKubeletVersionEnabled bool
}

func validateConfigNodeForExtremeLatencyProfile(obj, oldObj *configv1.Node) *field.Error {
	fromProfile := oldObj.Spec.WorkerLatencyProfile
	toProfile := obj.Spec.WorkerLatencyProfile

	for _, rejectionScenario := range rejectionScenarios {
		if fromProfile == rejectionScenario.fromProfile && toProfile == rejectionScenario.toProfile {
			return field.Invalid(field.NewPath("spec", "workerLatencyProfile"), obj.Spec.WorkerLatencyProfile,
				fmt.Sprintf(
					"cannot update worker latency profile from %q to %q as extreme profile transition is unsupported, please select any other profile with supported transition such as %q",
					oldObj.Spec.WorkerLatencyProfile,
					obj.Spec.WorkerLatencyProfile,
					configv1.MediumUpdateAverageReaction,
				),
			)
		}
	}
	return nil
}

func (c *configNodeV1) ValidateCreate(_ context.Context, uncastObj runtime.Object) field.ErrorList {
	obj, allErrs := toConfigNodeV1(uncastObj)
	if len(allErrs) > 0 {
		return allErrs
	}

	allErrs = append(allErrs, validation.ValidateObjectMeta(&obj.ObjectMeta, false, customresourcevalidation.RequireNameCluster, field.NewPath("metadata"))...)
	if err := c.validateMinimumKubeletVersion(obj); err != nil {
		allErrs = append(allErrs, err)
	}

	return allErrs
}

func (c *configNodeV1) ValidateUpdate(_ context.Context, uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
	obj, allErrs := toConfigNodeV1(uncastObj)
	if len(allErrs) > 0 {
		return allErrs
	}
	oldObj, allErrs := toConfigNodeV1(uncastOldObj)
	if len(allErrs) > 0 {
		return allErrs
	}

	allErrs = append(allErrs, validation.ValidateObjectMetaUpdate(&obj.ObjectMeta, &oldObj.ObjectMeta, field.NewPath("metadata"))...)
	if err := validateConfigNodeForExtremeLatencyProfile(obj, oldObj); err != nil {
		allErrs = append(allErrs, err)
	}
	if err := c.validateMinimumKubeletVersion(obj); err != nil {
		allErrs = append(allErrs, err)
	}

	return allErrs
}
func (c *configNodeV1) validateMinimumKubeletVersion(obj *configv1.Node) *field.Error {
	if !c.minimumKubeletVersionEnabled {
		return nil
	}
	fieldPath := field.NewPath("spec", "minimumKubeletVersion")
	if !c.waitForNodeInformerSyncedFn() {
		return field.InternalError(fieldPath, fmt.Errorf("caches not synchronized, cannot validate minimumKubeletVersion"))
	}

	nodes, err := c.nodeListerFn().List(labels.Everything())
	if err != nil {
		return field.NotFound(fieldPath, fmt.Sprintf("Getting nodes to compare minimum version %v", err.Error()))
	}

	if err := nodelib.ValidateMinimumKubeletVersion(nodes, obj.Spec.MinimumKubeletVersion); err != nil {
		if errors.Is(err, nodelib.ErrKubeletOutdated) {
			return field.Forbidden(fieldPath, err.Error())
		}
		return field.Invalid(fieldPath, obj.Spec.MinimumKubeletVersion, err.Error())
	}
	return nil
}

func (*configNodeV1) ValidateStatusUpdate(_ context.Context, uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
	obj, errs := toConfigNodeV1(uncastObj)
	if len(errs) > 0 {
		return errs
	}
	oldObj, errs := toConfigNodeV1(uncastOldObj)
	if len(errs) > 0 {
		return errs
	}

	// TODO validate the obj.  remember that status validation should *never* fail on spec validation errors.
	errs = append(errs, validation.ValidateObjectMetaUpdate(&obj.ObjectMeta, &oldObj.ObjectMeta, field.NewPath("metadata"))...)

	return errs
}

type configNodeV1Wrapper struct {
	// handler is only used to know if the plugin is ready to process requests.
	handler admission.Handler

	nodeLister corev1listers.NodeLister
	delegate   admission.ValidationInterface
}

var (
	_ = initializer.WantsExternalKubeInformerFactory(&configNodeV1Wrapper{})
	_ = admission.ValidationInterface(&configNodeV1Wrapper{})
)

func (c *configNodeV1Wrapper) SetExternalKubeInformerFactory(kubeInformers informers.SharedInformerFactory) {
	nodeInformer := kubeInformers.Core().V1().Nodes()
	c.nodeLister = nodeInformer.Lister()
	c.handler.SetReadyFunc(nodeInformer.Informer().HasSynced)
}

func (c *configNodeV1Wrapper) ValidateInitialization() error {
	if c.nodeLister == nil {
		return fmt.Errorf("%s needs a nodes lister", PluginName)
	}

	return nil
}

func (c *configNodeV1Wrapper) getNodeLister() corev1listers.NodeLister {
	return c.nodeLister
}

func (c *configNodeV1Wrapper) waitForNodeInformerSyncedFn() bool {
	return c.handler.WaitForReady()
}

func (c *configNodeV1Wrapper) Validate(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) (err error) {
	return c.delegate.Validate(ctx, a, o)
}

func (c *configNodeV1Wrapper) Handles(operation admission.Operation) bool {
	return c.delegate.Handles(operation)
}
