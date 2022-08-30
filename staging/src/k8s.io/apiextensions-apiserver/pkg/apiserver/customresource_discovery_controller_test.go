package apiserver

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset/fake"
	"k8s.io/apiextensions-apiserver/pkg/client/informers/externalversions"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/endpoints/discovery"
	v2 "k8s.io/apiserver/pkg/endpoints/discovery/v2"
)

var coolFooCRD = &v1.CustomResourceDefinition{
	TypeMeta: metav1.TypeMeta{
		APIVersion: "apiextensions.k8s.io/v1",
		Kind:       "CustomResourceDefinition",
	},
	ObjectMeta: metav1.ObjectMeta{
		Name: "coolfoo.stable.example.com",
	},
	Spec: v1.CustomResourceDefinitionSpec{
		Group: "stable.example.com",
		Names: v1.CustomResourceDefinitionNames{
			Plural:     "coolfoos",
			Singular:   "coolfoo",
			ShortNames: []string{"foo"},
			Kind:       "CoolFoo",
			ListKind:   "CoolFooList",
			Categories: []string{"cool"},
		},
		Scope: v1.ClusterScoped,
		Versions: []v1.CustomResourceDefinitionVersion{
			{
				Name:       "v1",
				Served:     true,
				Storage:    true,
				Deprecated: false,
				Subresources: &v1.CustomResourceSubresources{
					// This CRD has a /status subresource
					Status: &v1.CustomResourceSubresourceStatus{},
				},
				Schema: &v1.CustomResourceValidation{
					// Unused by discovery
					OpenAPIV3Schema: &v1.JSONSchemaProps{},
				},
			},
		},
		Conversion:            &v1.CustomResourceConversion{},
		PreserveUnknownFields: false,
	},
	Status: v1.CustomResourceDefinitionStatus{
		Conditions: []v1.CustomResourceDefinitionCondition{
			{
				Type:   v1.Established,
				Status: v1.ConditionTrue,
			},
		},
	},
}
var coolFooDiscovery metav1.APIGroupDiscovery = metav1.APIGroupDiscovery{
	TypeMeta: metav1.TypeMeta{},
	ObjectMeta: metav1.ObjectMeta{
		Name: "stable.example.com",
	},
	Versions: []metav1.APIVersionDiscovery{
		{
			Version: "v1",
			Resources: []metav1.APIResourceDiscovery{
				{
					Resource:     "coolfoos",
					Scope:        metav1.ScopeCluster,
					SingularName: "coolfoo",
					Verbs:        []string{"delete", "deletecollection", "get", "list", "patch", "create", "update", "watch"},
					ShortNames:   []string{"foo"},
					Categories:   []string{"cool"},
					ReturnType: metav1.APIDiscoveryKind{
						Group:   "stable.example.com",
						Version: "v1",
						Kind:    "CoolFoo",
					},
					Subresources: []metav1.APISubresourceDiscovery{
						{
							Subresource:   "status",
							Verbs:         []string{"get", "patch", "update"},
							AcceptedTypes: nil, // is this correct?
							ReturnType: &metav1.APIDiscoveryKind{
								Group:   "stable.example.com",
								Version: "v1",
								Kind:    "CoolFoo",
							},
						},
					},
				},
			},
		},
	},
	Status: metav1.APIGroupDiscoveryStatus{},
}

func init() {
	// Not testing against an apiserver, so just assume names are accepted
	coolFooCRD.Status.AcceptedNames = coolFooCRD.Spec.Names
}

// Provides an apiextensions-apiserver client
type testEnvironment struct {
	clientset.Interface

	// Discovery test details
	versionDiscoveryHandler
	groupDiscoveryHandler

	v2.FakeResourceManager
}

func (env *testEnvironment) Start(ctx context.Context) {
	discoverySyncedCh := make(chan struct{})

	factory := externalversions.NewSharedInformerFactoryWithOptions(
		env.Interface, 30*time.Second)

	discoveryController := NewDiscoveryController(
		factory.Apiextensions().V1().CustomResourceDefinitions(),
		&env.versionDiscoveryHandler,
		&env.groupDiscoveryHandler,
		env.FakeResourceManager,
	)

	factory.Start(ctx.Done())
	go discoveryController.Run(ctx.Done(), discoverySyncedCh)

	select {
	case <-discoverySyncedCh:
	case <-ctx.Done():
	}
}

func setup() *testEnvironment {
	env := &testEnvironment{
		Interface:           fake.NewSimpleClientset(),
		FakeResourceManager: v2.NewFakeResourceManager(),
		versionDiscoveryHandler: versionDiscoveryHandler{
			discovery: make(map[schema.GroupVersion]*discovery.APIVersionHandler),
		},
		groupDiscoveryHandler: groupDiscoveryHandler{
			discovery: make(map[string]*discovery.APIGroupHandler),
		},
	}

	return env
}

func TestResourceManagerExistingCRD(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	env := setup()
	_, err := env.Interface.
		ApiextensionsV1().
		CustomResourceDefinitions().
		Create(
			ctx,
			coolFooCRD,
			metav1.CreateOptions{
				FieldManager: "resource-manager-test",
			},
		)

	require.NoError(t, err)

	env.FakeResourceManager.Expect().
		AddGroupVersion(coolFooCRD.Spec.Group, coolFooDiscovery.Versions[0])

	env.Start(ctx)
	require.NoError(t, env.FakeResourceManager.Validate())
}

// Tests that if a CRD is added a runtime, the discovery controller will
// put its information in the discovery document
func TestResourceManagerAddedCRD(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	env := setup()
	env.FakeResourceManager.Expect().
		AddGroupVersion(coolFooCRD.Spec.Group, coolFooDiscovery.Versions[0])

	env.Start(ctx)

	// Create CRD after the controller has already started
	_, err := env.Interface.
		ApiextensionsV1().
		CustomResourceDefinitions().
		Create(
			ctx,
			coolFooCRD,
			metav1.CreateOptions{
				FieldManager: "resource-manager-test",
			},
		)

	require.NoError(t, err)

	err = env.FakeResourceManager.WaitForActions(ctx, 1*time.Second)
	require.NoError(t, err)
}

// Tests that if a CRD is deleted at runtime, the discovery controller will
// remove its information from its ResourceManager
func TestDiscoveryControllerResourceManagerRemovedCRD(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	env := setup()
	env.Start(ctx)

	// Create CRD after the controller has already started
	_, err := env.Interface.
		ApiextensionsV1().
		CustomResourceDefinitions().
		Create(
			ctx,
			coolFooCRD,
			metav1.CreateOptions{},
		)

	require.NoError(t, err)

	// Wait for the Controller to pick up the Create event and add it to the
	// Resource Manager
	env.FakeResourceManager.Expect().
		AddGroupVersion(coolFooCRD.Spec.Group, coolFooDiscovery.Versions[0])
	err = env.FakeResourceManager.WaitForActions(ctx, 1*time.Second)
	require.NoError(t, err)

	err = env.Interface.
		ApiextensionsV1().
		CustomResourceDefinitions().
		Delete(ctx, coolFooCRD.Name, metav1.DeleteOptions{})

	require.NoError(t, err)

	// Wait for the Controller to detect there are no more CRDs of this group
	// and remove the entire group
	env.FakeResourceManager.Expect().RemoveGroup(coolFooCRD.Spec.Group)

	err = env.FakeResourceManager.WaitForActions(ctx, 1*time.Second)
	require.NoError(t, err)
}
