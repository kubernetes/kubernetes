package imagedigestmirrorset

import (
	"testing"

	configv1 "github.com/openshift/api/config/v1"
	operatorsv1alpha1 "github.com/openshift/api/operator/v1alpha1"
	configclientfake "github.com/openshift/client-go/config/clientset/versioned/fake"
	operatorclientfake "github.com/openshift/client-go/operator/clientset/versioned/fake"
	operatorsv1alpha1client "github.com/openshift/client-go/operator/clientset/versioned/typed/operator/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestValidateIDMSUse(t *testing.T) {
	for _, tt := range []struct {
		name           string
		objects        []runtime.Object
		idmss          []runtime.Object
		action         string
		validateErrors func(t *testing.T, errs field.ErrorList)
	}{
		{
			name: "can't create idms with existing icsp",
			objects: []runtime.Object{
				&operatorsv1alpha1.ImageContentSourcePolicy{
					ObjectMeta: metav1.ObjectMeta{Name: "icsp"},
					Spec: operatorsv1alpha1.ImageContentSourcePolicySpec{
						RepositoryDigestMirrors: []operatorsv1alpha1.RepositoryDigestMirrors{
							{Source: "example.com", Mirrors: []string{"mirror.com"}},
						},
					},
				},
			},
			action: "create",

			validateErrors: func(t *testing.T, errs field.ErrorList) {
				t.Helper()
				if len(errs) != 1 {
					t.Fatal(errs)
				}
				if errs[0].Error() != `Kind.ImageDigestMirrorSet: Forbidden: can't create ImageDigestMirrorSet when ImageContentSourcePolicy resources exist` {
					t.Error(errs[0])
				}
			},
		},

		{
			name:   "success creating idms",
			action: "create",
			validateErrors: func(t *testing.T, errs field.ErrorList) {
				t.Helper()
				if len(errs) > 0 {
					t.Fatal(errs)
				}
			},
		},
		{
			name:   "success updating idms",
			action: "update",
			idmss: []runtime.Object{
				&configv1.ImageDigestMirrorSet{
					ObjectMeta: metav1.ObjectMeta{Name: "idms"},
					Spec: configv1.ImageDigestMirrorSetSpec{
						ImageDigestMirrors: []configv1.ImageDigestMirrors{
							{Source: "example.com", Mirrors: []configv1.ImageMirror{"mirror.com"}},
						},
					},
				},
			},
			validateErrors: func(t *testing.T, errs field.ErrorList) {
				t.Helper()
				if len(errs) > 0 {
					t.Fatal(errs)
				}
			},
		},
		{
			name:   "success updating idms status",
			action: "update",
			idmss: []runtime.Object{
				&configv1.ImageDigestMirrorSet{
					ObjectMeta: metav1.ObjectMeta{Name: "idms"},
					Spec: configv1.ImageDigestMirrorSetSpec{
						ImageDigestMirrors: []configv1.ImageDigestMirrors{
							{Source: "example.com", Mirrors: []configv1.ImageMirror{"mirror.com"}},
						},
					},
					Status: configv1.ImageDigestMirrorSetStatus{},
				},
			},
			validateErrors: func(t *testing.T, errs field.ErrorList) {
				t.Helper()
				if len(errs) > 0 {
					t.Fatal(errs)
				}
			},
		},
		{
			name:   "success updating idms with itms",
			action: "update",
			idmss: []runtime.Object{
				&configv1.ImageTagMirrorSet{
					ObjectMeta: metav1.ObjectMeta{Name: "itms"},
					Spec: configv1.ImageTagMirrorSetSpec{
						ImageTagMirrors: []configv1.ImageTagMirrors{
							{Source: "example.com", Mirrors: []configv1.ImageMirror{"mirror.com"}},
						},
					},
				},
				&configv1.ImageDigestMirrorSet{
					ObjectMeta: metav1.ObjectMeta{Name: "idms"},
					Spec: configv1.ImageDigestMirrorSetSpec{
						ImageDigestMirrors: []configv1.ImageDigestMirrors{
							{Source: "example.com", Mirrors: []configv1.ImageMirror{"mirror.com"}},
						},
					},
				},
			},
			validateErrors: func(t *testing.T, errs field.ErrorList) {
				t.Helper()
				if len(errs) > 0 {
					t.Fatal(errs)
				}
			},
		},
		{
			// this case is technically impossible, we prohibit to have idms to be updated when there is icsp
			name:   "updating idms with icsp",
			action: "update",
			objects: []runtime.Object{
				&operatorsv1alpha1.ImageContentSourcePolicy{
					ObjectMeta: metav1.ObjectMeta{Name: "icsp"},
					Spec: operatorsv1alpha1.ImageContentSourcePolicySpec{
						RepositoryDigestMirrors: []operatorsv1alpha1.RepositoryDigestMirrors{
							{Source: "example.com", Mirrors: []string{"mirror.com"}},
						},
					},
				},
			},
			idmss: []runtime.Object{
				&configv1.ImageDigestMirrorSet{
					ObjectMeta: metav1.ObjectMeta{Name: "idms"},
					Spec: configv1.ImageDigestMirrorSetSpec{
						ImageDigestMirrors: []configv1.ImageDigestMirrors{
							{Source: "example.com", Mirrors: []configv1.ImageMirror{"mirror.com"}},
						},
					},
				},
			},
			validateErrors: func(t *testing.T, errs field.ErrorList) {
				t.Helper()
				if len(errs) != 1 {
					t.Fatal(errs)
				}
				if errs[0].Error() != `Kind.ImageDigestMirrorSet: Forbidden: can't update ImageDigestMirrorSet when ImageContentSourcePolicy resources exist` {
					t.Error(errs[0])
				}
			},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			fakeclient := operatorclientfake.NewSimpleClientset(tt.objects...)
			_ = configclientfake.NewSimpleClientset(tt.idmss...)
			instance := imagedigestmirrorsetV1{
				imageContentSourcePoliciesGetter: func() operatorsv1alpha1client.ImageContentSourcePoliciesGetter {
					return fakeclient.OperatorV1alpha1()
				},
			}
			tt.validateErrors(t, ValidateITMSIDMSUse(tt.action, instance.imageContentSourcePoliciesGetter(), IDMSKind))
		})
	}

}
