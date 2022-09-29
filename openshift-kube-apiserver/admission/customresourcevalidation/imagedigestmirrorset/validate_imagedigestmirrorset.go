package imagedigestmirrorset

import (
	"context"
	"fmt"
	"io"

	v1 "github.com/openshift/api/config/v1"
	operatorsv1alpha1client "github.com/openshift/client-go/operator/clientset/versioned/typed/operator/v1alpha1"
	"github.com/openshift/library-go/pkg/apiserver/admission/admissionrestconfig"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/client-go/rest"

	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation"
)

const (
	PluginName   = "config.openshift.io/ValidateImageDigestMirrorSet"
	ICSPKind     = "ImageContentSourcePolicy"
	ICSPResource = "imagecontentsourcepolicies"
	IDMSKind     = "ImageDigestMirrorSet"
	IDMSResource = "imagedigestmirrorsets"
	ITMSKind     = "ImageTagMirrorSet"
	ITMSResource = "imagetagmirrorsets"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {

	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return newValidateIDMS()
	})
}

type validateCustomResourceWithClient struct {
	admission.ValidationInterface

	imageContentSourcePoliciesGetter operatorsv1alpha1client.ImageContentSourcePoliciesGetter
}

func newValidateIDMS() (admission.Interface, error) {
	ret := &validateCustomResourceWithClient{}

	delegate, err := customresourcevalidation.NewValidator(
		map[schema.GroupResource]bool{
			v1.Resource(IDMSResource): true,
		},
		map[schema.GroupVersionKind]customresourcevalidation.ObjectValidator{
			v1.GroupVersion.WithKind(IDMSKind): imagedigestmirrorsetV1{imageContentSourcePoliciesGetter: ret.getImageContentSourcePoliciesGetter},
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

type imagedigestmirrorsetV1 struct {
	imageContentSourcePoliciesGetter func() operatorsv1alpha1client.ImageContentSourcePoliciesGetter
}

func (i imagedigestmirrorsetV1) ValidateCreate(uncastObj runtime.Object) field.ErrorList {
	return ValidateITMSIDMSUse("create", i.imageContentSourcePoliciesGetter(), IDMSKind)
}

func (i imagedigestmirrorsetV1) ValidateUpdate(uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
	return ValidateITMSIDMSUse("update", i.imageContentSourcePoliciesGetter(), IDMSKind)
}

func (i imagedigestmirrorsetV1) ValidateStatusUpdate(uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
	return ValidateITMSIDMSUse("update", i.imageContentSourcePoliciesGetter(), IDMSKind)
}

func ValidateITMSIDMSUse(action string, icspGetter operatorsv1alpha1client.ImageContentSourcePoliciesGetter, kind string) field.ErrorList {
	icspList, err := icspGetter.ImageContentSourcePolicies().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return field.ErrorList{field.InternalError(field.NewPath("Kind", ICSPKind), err)}
	}
	if len(icspList.Items) > 0 {
		return field.ErrorList{
			field.Forbidden(field.NewPath("Kind", kind), fmt.Sprintf("can't %s %s when %s resources exist", action, kind, ICSPKind))}
	}
	return nil
}
