package securitycontextconstraints

import (
	"bytes"
	"context"
	"io"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/admission"

	securityv1 "github.com/openshift/api/security/v1"
)

const DefaultingPluginName = "security.openshift.io/DefaultSecurityContextConstraints"

func RegisterDefaulting(plugins *admission.Plugins) {
	plugins.Register(DefaultingPluginName, func(config io.Reader) (admission.Interface, error) {
		return NewDefaulter(), nil
	})
}

type defaultSCC struct {
	*admission.Handler

	scheme       *runtime.Scheme
	codecFactory runtimeserializer.CodecFactory
}

var _ admission.MutationInterface = &defaultSCC{}

func NewDefaulter() admission.Interface {
	scheme := runtime.NewScheme()
	codecFactory := runtimeserializer.NewCodecFactory(scheme)
	utilruntime.Must(securityv1.Install(scheme))

	return &defaultSCC{
		Handler:      admission.NewHandler(admission.Create, admission.Update),
		scheme:       scheme,
		codecFactory: codecFactory,
	}
}

// Admit defaults an SCC by going unstructured > external > internal > external > unstructured
func (a *defaultSCC) Admit(ctx context.Context, attributes admission.Attributes, o admission.ObjectInterfaces) error {
	if a.shouldIgnore(attributes) {
		return nil
	}

	unstructuredOrig, ok := attributes.GetObject().(*unstructured.Unstructured)
	if !ok {
		return nil
	}
	buf := &bytes.Buffer{}
	if err := unstructured.UnstructuredJSONScheme.Encode(unstructuredOrig, buf); err != nil {
		return err
	}

	uncastObj, err := runtime.Decode(a.codecFactory.UniversalDeserializer(), buf.Bytes())
	if err != nil {
		return err
	}

	outSCCExternal := uncastObj.(*securityv1.SecurityContextConstraints)
	SetDefaults_SCC(outSCCExternal)
	defaultedBytes, err := runtime.Encode(a.codecFactory.LegacyCodec(securityv1.GroupVersion), outSCCExternal)
	if err != nil {
		return err
	}
	outUnstructured := &unstructured.Unstructured{}
	if _, _, err := unstructured.UnstructuredJSONScheme.Decode(defaultedBytes, nil, outUnstructured); err != nil {
		return err
	}

	unstructuredOrig.Object = outUnstructured.Object

	return nil
}

func (a *defaultSCC) shouldIgnore(attributes admission.Attributes) bool {
	if attributes.GetResource().GroupResource() != (schema.GroupResource{Group: "security.openshift.io", Resource: "securitycontextconstraints"}) {
		return true
	}
	// if a subresource is specified, skip it
	if len(attributes.GetSubresource()) > 0 {
		return true
	}

	return false
}
