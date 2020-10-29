package route

import (
	"context"
	"testing"

	routev1 "github.com/openshift/api/route/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/client-go/kubernetes/fake"
)

// setupWithFakeClient setter is only available in unit-tests
func (a *validateCustomResourceWithClient) setupWithFakeClient() {
	c := fake.NewSimpleClientset()
	a.secretsGetter = c.CoreV1()
	a.sarGetter = c.AuthorizationV1()
	a.routeValidationOptsGetter = NewRouteValidationOpts()
}

// TestValidateRoutePlugin verifies if the route validation plugin can handle admits
// for the resource {group: api/route/v1, kind: Route}
// will check if validator client is
// conformant with admission.InitializationValidator interface
func TestValidateRoutePlugin(t *testing.T) {
	plugin, err := NewValidateRoute()
	if err != nil {
		t.Fatal(err)
	}

	validator, ok := plugin.(*validateCustomResourceWithClient)
	if !ok {
		t.Fatal("could not type cast returned value of NewValidateRoute() into type validateCustomResourceWithClient, " +
			"perhaps you changed the type in the implementation but not in the tests!")
	}

	// unit test specific logic as a replacement for routeAdmitter.SetRESTClientConfig(...)
	validator.setupWithFakeClient()

	// admission.InitializationValidator -> ValidateInitialization()
	err = validator.ValidateInitialization()
	if err != nil {
		t.Fatal(err)
	}

	r1 := &routev1.Route{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: "bar",
		},
		Spec: routev1.RouteSpec{
			To: routev1.RouteTargetReference{
				Kind: "Service",
				Name: "default",
			},
		},
	}
	r2 := r1.DeepCopy()

	s1 := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: "bar",
		},
		Data: map[string][]byte{},
	}
	s2 := s1.DeepCopy()

	testCases := []struct {
		description string

		object    runtime.Object
		oldObject runtime.Object

		kind     schema.GroupVersionKind
		resource schema.GroupVersionResource

		name      string
		namespace string

		expectedError bool
	}{
		{
			description: "route object is passed to admission plugin with scheme routev1.Route",

			object:    runtime.Object(r1),
			oldObject: runtime.Object(r2),

			kind:     routev1.GroupVersion.WithKind("Route"),
			resource: routev1.GroupVersion.WithResource("routes"),

			name:      r1.Name,
			namespace: r1.Namespace,

			expectedError: false,
		},
		{
			description: "non-route object is passed to admission plugin with scheme corev1.Secret",

			object:    runtime.Object(s1),
			oldObject: runtime.Object(s2),

			kind:     corev1.SchemeGroupVersion.WithKind("Secret"),
			resource: corev1.SchemeGroupVersion.WithResource("secrets"),

			name:      s1.Name,
			namespace: s1.Namespace,

			expectedError: false,
		},
		{
			description: "non-route object is passed to admission plugin with conflicting scheme routev1.Route",

			object:    runtime.Object(s1),
			oldObject: runtime.Object(s2),

			kind:     routev1.GroupVersion.WithKind("Route"),
			resource: routev1.GroupVersion.WithResource("routes"),

			name:      s1.Name,
			namespace: s1.Namespace,

			expectedError: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {

			attr := admission.NewAttributesRecord(
				tc.object, tc.oldObject,
				tc.kind, tc.name, tc.namespace, tc.resource,
				"", admission.Create, nil, false,
				&user.DefaultInfo{},
			)

			switch err := validator.Validate(context.Background(), attr, nil); {
			case !tc.expectedError && err != nil:
				t.Fatalf("admission error not expected, but found %q", err)
			case tc.expectedError && err == nil:
				t.Fatal("admission error expected, but got nil")
			}
		})
	}
}
