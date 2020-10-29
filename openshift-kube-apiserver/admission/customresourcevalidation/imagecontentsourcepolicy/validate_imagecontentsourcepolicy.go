package imagecontentsourcepolicy

import (
	"context"
	"fmt"
	"io"

	operatorv1alpha1 "github.com/openshift/api/operator/v1alpha1"
	configv1client "github.com/openshift/client-go/config/clientset/versioned/typed/config/v1"
	"github.com/openshift/library-go/pkg/apiserver/admission/admissionrestconfig"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/client-go/rest"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/imagedigestmirrorset"
)

const PluginName = "operator.openshift.io/ValidateImageContentSourcePolicy"

// Register registers a plugin
func Register(plugins *admission.Plugins) {

	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return newValidateICSP()
	})
}

type validateCustomResourceWithClient struct {
	admission.ValidationInterface

	imageDigestMirrorSetsGetter configv1client.ImageDigestMirrorSetsGetter
	imageTagMirrorSetsGetter    configv1client.ImageTagMirrorSetsGetter
}

func newValidateICSP() (admission.Interface, error) {
	ret := &validateCustomResourceWithClient{}

	delegate, err := customresourcevalidation.NewValidator(
		map[schema.GroupResource]bool{
			operatorv1alpha1.Resource(imagedigestmirrorset.ICSPResource): true,
		},
		map[schema.GroupVersionKind]customresourcevalidation.ObjectValidator{
			operatorv1alpha1.GroupVersion.WithKind(imagedigestmirrorset.ICSPKind): imagecontentsourcepolicy{imageDigestMirrorSetsGetter: ret.getImageDigestMirrorSetsGetter, imageTagMirrorSetsGetter: ret.getImageTagMirrorSetsGetter},
		})
	if err != nil {
		return nil, err
	}
	ret.ValidationInterface = delegate
	return ret, nil
}

var _ admissionrestconfig.WantsRESTClientConfig = &validateCustomResourceWithClient{}

func (i *validateCustomResourceWithClient) getImageDigestMirrorSetsGetter() configv1client.ImageDigestMirrorSetsGetter {
	return i.imageDigestMirrorSetsGetter
}

func (i *validateCustomResourceWithClient) getImageTagMirrorSetsGetter() configv1client.ImageTagMirrorSetsGetter {
	return i.imageTagMirrorSetsGetter
}

func (i *validateCustomResourceWithClient) SetRESTClientConfig(restClientConfig rest.Config) {
	client, err := configv1client.NewForConfig(&restClientConfig)
	if err != nil {
		utilruntime.HandleError(err)
		return
	}
	i.imageDigestMirrorSetsGetter = client
	i.imageTagMirrorSetsGetter = client
}

func (i *validateCustomResourceWithClient) ValidateInitialization() error {
	if i.imageDigestMirrorSetsGetter == nil {
		return fmt.Errorf(PluginName + " needs an imageDigestMirrorSetsGetter")
	}

	if i.imageTagMirrorSetsGetter == nil {
		return fmt.Errorf(PluginName + " needs an imageTagMirrorSetsGetter")
	}

	if initializationValidator, ok := i.ValidationInterface.(admission.InitializationValidator); ok {
		return initializationValidator.ValidateInitialization()
	}

	return nil
}

type imagecontentsourcepolicy struct {
	imageDigestMirrorSetsGetter func() configv1client.ImageDigestMirrorSetsGetter
	imageTagMirrorSetsGetter    func() configv1client.ImageTagMirrorSetsGetter
}

func (i imagecontentsourcepolicy) ValidateCreate(uncastObj runtime.Object) field.ErrorList {
	return i.validateICSPUse("create")
}

func (i imagecontentsourcepolicy) ValidateUpdate(uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
	return i.validateICSPUse("update")
}

func (i imagecontentsourcepolicy) ValidateStatusUpdate(uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
	return i.validateICSPUse("update")
}

func (i imagecontentsourcepolicy) validateICSPUse(action string) field.ErrorList {
	idmsList, err := i.imageDigestMirrorSetsGetter().ImageDigestMirrorSets().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return field.ErrorList{field.InternalError(field.NewPath("Kind", imagedigestmirrorset.IDMSKind), err)}
	}
	if len(idmsList.Items) > 0 {
		return field.ErrorList{
			field.Forbidden(field.NewPath("Kind", imagedigestmirrorset.ICSPKind), fmt.Sprintf("can't %s %s when %s resources exist", action, imagedigestmirrorset.ICSPKind, imagedigestmirrorset.IDMSKind))}
	}
	itmsList, err := i.imageTagMirrorSetsGetter().ImageTagMirrorSets().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return field.ErrorList{field.InternalError(field.NewPath("Kind", imagedigestmirrorset.ITMSKind), err)}
	}
	if len(itmsList.Items) > 0 {
		return field.ErrorList{
			field.Forbidden(field.NewPath("Kind", imagedigestmirrorset.ICSPKind), fmt.Sprintf("can't %s %s when %s resources exist", action, imagedigestmirrorset.ICSPKind, imagedigestmirrorset.ITMSKind))}
	}
	return nil
}
