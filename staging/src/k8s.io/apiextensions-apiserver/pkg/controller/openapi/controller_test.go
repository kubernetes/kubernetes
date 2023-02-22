package openapi

import (
	"context"
	"testing"
	"time"

	"k8s.io/apiextensions-apiserver/pkg/client/informers/externalversions"

	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset/fake"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kube-openapi/pkg/validation/spec"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/util/wait"
)

func NewFakeOpenAPIService() *FakeOpenAPIService {
	return &FakeOpenAPIService{expectedUpdates: 0}
}

type FakeOpenAPIService struct {
	updates         []*spec.Swagger
	expectedUpdates int
}

func (o *FakeOpenAPIService) UpdateSpec(s *spec.Swagger) error {
	o.updates = append(o.updates, s)
	return nil
}

func (o *FakeOpenAPIService) ExpectUpdate() {
	o.expectedUpdates += 1
}

func (o *FakeOpenAPIService) GetLastOpenAPI() *spec.Swagger {
	if len(o.updates) == 0 {
		return nil
	}
	return o.updates[len(o.updates)-1]
}

func (o *FakeOpenAPIService) WaitForActions() error {
	err := wait.PollImmediate(
		100*time.Millisecond,
		3*time.Second,
		func() (done bool, err error) {
			if len(o.updates) >= o.expectedUpdates {
				return true, nil
			}
			return false, nil
		})
	return err
}

var coolFooCRDPath = "/apis/structural.cr.bar.com/v1/foos"
var coolFooCRD = &apiextensionsv1.CustomResourceDefinition{
	ObjectMeta: metav1.ObjectMeta{
		Name: "foos.structural.cr.bar.com",
	},
	Spec: apiextensionsv1.CustomResourceDefinitionSpec{
		Group: "structural.cr.bar.com",
		Scope: apiextensionsv1.NamespaceScoped,
		Names: apiextensionsv1.CustomResourceDefinitionNames{
			Plural: "foos",
			Kind:   "Foo",
		},
		Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
			{
				Name:    "v1",
				Served:  true,
				Storage: true,
				Schema: &apiextensionsv1.CustomResourceValidation{
					OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
						Type: "object",
						Properties: map[string]apiextensionsv1.JSONSchemaProps{
							"foo": {Type: "string"},
						},
					},
				},
			},
		},
	},
	Status: apiextensionsv1.CustomResourceDefinitionStatus{
		Conditions: []apiextensionsv1.CustomResourceDefinitionCondition{
			{
				Type:   apiextensionsv1.Established,
				Status: apiextensionsv1.ConditionTrue,
			},
		},
	},
}

func TestBasicCRD(t *testing.T) {
	fakeClient := fake.NewSimpleClientset()
	factory := externalversions.NewSharedInformerFactoryWithOptions(
		fakeClient, 30*time.Second)

	controller := NewController(factory.Apiextensions().V1().CustomResourceDefinitions())
	f := NewFakeOpenAPIService()
	stopCh := make(chan struct{})
	defer close(stopCh)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	factory.Start(ctx.Done())
	go controller.Run(&spec.Swagger{}, f, stopCh)
	f.ExpectUpdate()
	if err := f.WaitForActions(); err != nil {
		t.Error(err)
	}

	_, err := fakeClient.ApiextensionsV1().CustomResourceDefinitions().Create(ctx, coolFooCRD, metav1.CreateOptions{FieldManager: "resource-manager-test"})
	if err != nil {
		t.Error(err)
	}
	f.ExpectUpdate()
	if err := f.WaitForActions(); err != nil {
		t.Error(err)
	}
	swagger := f.GetLastOpenAPI()
	_, ok := swagger.Paths.Paths[coolFooCRDPath]
	if !ok {
		t.Errorf("Expected path %s after applying CRD", coolFooCRDPath)
	}

}
