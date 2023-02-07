package imagetagmirrorset

import (
	"fmt"
	"io"

	v1 "github.com/openshift/api/config/v1"
	operatorsv1alpha1client "github.com/openshift/client-go/operator/clientset/versioned/typed/operator/v1alpha1"
	"github.com/openshift/library-go/pkg/apiserver/admission/admissionrestconfig"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/client-go/rest"

	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/imagedigestmirrorset"
)

const PluginName = "config.openshift.io/ValidateImageTagMirrorSet"

// Register registers a plugin
func Register(plugins *admission.Plugins) {

	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return newValidateITMS()
	})
}

type validateCustomResourceWithClient struct {
	admission.ValidationInterface

	imageContentSourcePoliciesGetter operatorsv1alpha1client.ImageContentSourcePoliciesGetter
}

func newValidateITMS() (admission.Interface, error) {
	ret := &validateCustomResourceWithClient{}

	delegate, err := customresourcevalidation.NewValidator(
		map[schema.GroupResource]bool{
			v1.Resource(imagedigestmirrorset.ITMSResource): true,
		},
		map[schema.GroupVersionKind]customresourcevalidation.ObjectValidator{
			v1.GroupVersion.WithKind(imagedigestmirrorset.ITMSKind): imagetagmirrorsetV1{imageContentSourcePoliciesGetter: ret.getImageContentSourcePoliciesGetter},
		})
	if err != nil {
		return nil, err
	}
	ret.ValidationInterface = delegate
	return ret, nil
}

var _ admissionrestconfig.WantsRESTClientConfig = &validateCustomResourceWithClient{}

func (i *validateCustomResourceWithClient) getImageContentSourcePoliciesGetter() operatorsv1alpha1client.ImageContentSourcePoliciesGetter {
	return i.imageContentSourcePoliciesGetter
}

func (i *validateCustomResourceWithClient) SetRESTClientConfig(restClientConfig rest.Config) {
	var err error
	i.imageContentSourcePoliciesGetter, err = operatorsv1alpha1client.NewForConfig(&restClientConfig)
	if err != nil {
		utilruntime.HandleError(err)
		return
	}
}

func (i *validateCustomResourceWithClient) ValidateInitialization() error {
	if i.imageContentSourcePoliciesGetter == nil {
		return fmt.Errorf(PluginName + " needs an imageContentSourcePoliciesGetter")
	}
	if initializationValidator, ok := i.ValidationInterface.(admission.InitializationValidator); ok {
		return initializationValidator.ValidateInitialization()
	}
	return nil
}

type imagetagmirrorsetV1 struct {
	imageContentSourcePoliciesGetter func() operatorsv1alpha1client.ImageContentSourcePoliciesGetter
}

func (i imagetagmirrorsetV1) ValidateCreate(uncastObj runtime.Object) field.ErrorList {
	return imagedigestmirrorset.ValidateITMSIDMSUse("create", i.imageContentSourcePoliciesGetter(), imagedigestmirrorset.ITMSKind)
}

func (i imagetagmirrorsetV1) ValidateUpdate(uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
	return imagedigestmirrorset.ValidateITMSIDMSUse("update", i.imageContentSourcePoliciesGetter(), imagedigestmirrorset.ITMSKind)
}

func (i imagetagmirrorsetV1) ValidateStatusUpdate(uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
	return imagedigestmirrorset.ValidateITMSIDMSUse("update", i.imageContentSourcePoliciesGetter(), imagedigestmirrorset.ITMSKind)
}
