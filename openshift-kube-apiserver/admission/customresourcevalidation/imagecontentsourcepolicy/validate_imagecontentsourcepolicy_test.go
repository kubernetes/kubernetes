package imagecontentsourcepolicy

import (
	"testing"

	configv1 "github.com/openshift/api/config/v1"
	operatorv1alpha1 "github.com/openshift/api/operator/v1alpha1"
	configclientfake "github.com/openshift/client-go/config/clientset/versioned/fake"
	configv1client "github.com/openshift/client-go/config/clientset/versioned/typed/config/v1"
	operatorclientfake "github.com/openshift/client-go/operator/clientset/versioned/fake"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestValidateICSPUse(t *testing.T) {

	for _, tt := range []struct {
		name           string
		objects        []runtime.Object
		icsps          []runtime.Object
		action         string
		validateErrors func(t *testing.T, errs field.ErrorList)
	}{
		{
			name: "can't create icsp with existing idms",
			objects: []runtime.Object{
				&configv1.ImageDigestMirrorSet{
					ObjectMeta: metav1.ObjectMeta{Name: "idms"},
					Spec: configv1.ImageDigestMirrorSetSpec{
						ImageDigestMirrors: []configv1.ImageDigestMirrors{
							{Source: "example.com", Mirrors: []configv1.ImageMirror{"mirror.com"}},
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
				if errs[0].Error() != `Kind.ImageContentSourcePolicy: Forbidden: can't create ImageContentSourcePolicy when ImageDigestMirrorSet resources exist` {
					t.Error(errs[0])
				}
			},
		},
		{
			name: "can't create icsp with existing itms",
			objects: []runtime.Object{
				&configv1.ImageTagMirrorSet{
					ObjectMeta: metav1.ObjectMeta{Name: "itms"},
					Spec: configv1.ImageTagMirrorSetSpec{
						ImageTagMirrors: []configv1.ImageTagMirrors{
							{Source: "example.com", Mirrors: []configv1.ImageMirror{"mirror.com"}},
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
				if errs[0].Error() != `Kind.ImageContentSourcePolicy: Forbidden: can't create ImageContentSourcePolicy when ImageTagMirrorSet resources exist` {
					t.Error(errs[0])
				}
			},
		},
		{
			name:   "success creating icsp",
			action: "create",

			validateErrors: func(t *testing.T, errs field.ErrorList) {
				t.Helper()
				if len(errs) > 0 {
					t.Fatal(errs)
				}
			},
		},
		{
			name: "update existing icsp",
			icsps: []runtime.Object{
				&operatorv1alpha1.ImageContentSourcePolicy{
					ObjectMeta: metav1.ObjectMeta{Name: "icsp"},
					Spec: operatorv1alpha1.ImageContentSourcePolicySpec{
						RepositoryDigestMirrors: []operatorv1alpha1.RepositoryDigestMirrors{
							{Source: "example.com", Mirrors: []string{"mirror.com"}},
						},
					},
				},
			},
			action: "update",
			validateErrors: func(t *testing.T, errs field.ErrorList) {
				t.Helper()
				if len(errs) > 0 {
					t.Fatal(errs)
				}
			},
		},
		{
			// this case is technically impossible, we prohibit to have icsp to be updated when there is idms
			name: "update existing icsp with idms",
			objects: []runtime.Object{
				&configv1.ImageDigestMirrorSet{
					ObjectMeta: metav1.ObjectMeta{Name: "idms"},
					Spec: configv1.ImageDigestMirrorSetSpec{
						ImageDigestMirrors: []configv1.ImageDigestMirrors{
							{Source: "example.com", Mirrors: []configv1.ImageMirror{"mirror.com"}},
						},
					},
				},
			},
			icsps: []runtime.Object{
				&operatorv1alpha1.ImageContentSourcePolicy{
					ObjectMeta: metav1.ObjectMeta{Name: "icsp"},
					Spec: operatorv1alpha1.ImageContentSourcePolicySpec{
						RepositoryDigestMirrors: []operatorv1alpha1.RepositoryDigestMirrors{
							{Source: "example.com", Mirrors: []string{"mirror.com"}},
						},
					},
				},
			},
			action: "update",

			validateErrors: func(t *testing.T, errs field.ErrorList) {
				t.Helper()
				if len(errs) != 1 {
					t.Fatal(errs)
				}
				if errs[0].Error() != `Kind.ImageContentSourcePolicy: Forbidden: can't update ImageContentSourcePolicy when ImageDigestMirrorSet resources exist` {
					t.Error(errs[0])
				}
			},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			fakeclient := configclientfake.NewSimpleClientset(tt.objects...)
			_ = operatorclientfake.NewSimpleClientset(tt.icsps...)
			instance := imagecontentsourcepolicy{
				imageDigestMirrorSetsGetter: func() configv1client.ImageDigestMirrorSetsGetter {
					return fakeclient.ConfigV1()
				},
				imageTagMirrorSetsGetter: func() configv1client.ImageTagMirrorSetsGetter {
					return fakeclient.ConfigV1()
				},
			}
			tt.validateErrors(t, instance.validateICSPUse(tt.action))
		})
	}
}
