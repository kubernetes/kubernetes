package apiserver

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"

	configv1 "github.com/openshift/api/config/v1"
	configclientfake "github.com/openshift/client-go/config/clientset/versioned/fake"
	configv1client "github.com/openshift/client-go/config/clientset/versioned/typed/config/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestValidateSNINames(t *testing.T) {
	expectNoErrors := func(t *testing.T, errs field.ErrorList) {
		t.Helper()
		if len(errs) > 0 {
			t.Fatal(errs)
		}
	}

	tests := []struct {
		name string

		internalName string
		apiserver    *configv1.APIServer

		validateErrors func(t *testing.T, errs field.ErrorList)
	}{
		{
			name:           "no sni",
			internalName:   "internal.host.com",
			apiserver:      &configv1.APIServer{},
			validateErrors: expectNoErrors,
		},
		{
			name:         "allowed sni",
			internalName: "internal.host.com",
			apiserver: &configv1.APIServer{
				Spec: configv1.APIServerSpec{
					ServingCerts: configv1.APIServerServingCerts{
						NamedCertificates: []configv1.APIServerNamedServingCert{
							{
								Names: []string{"external.host.com", "somwhere.else.*"},
							},
						},
					},
				},
			},
			validateErrors: expectNoErrors,
		},
		{
			name:         "directly invalid sni",
			internalName: "internal.host.com",
			apiserver: &configv1.APIServer{
				Spec: configv1.APIServerSpec{
					ServingCerts: configv1.APIServerServingCerts{
						NamedCertificates: []configv1.APIServerNamedServingCert{
							{Names: []string{"external.host.com", "somwhere.else.*"}},
							{Names: []string{"foo.bar", "internal.host.com"}},
						},
					},
				},
			},
			validateErrors: func(t *testing.T, errs field.ErrorList) {
				t.Helper()
				if len(errs) != 1 {
					t.Fatal(errs)
				}
				if errs[0].Error() != `spec.servingCerts[1].names[1]: Invalid value: "internal.host.com": may not match internal loadbalancer: "internal.host.com"` {
					t.Error(errs[0])
				}
			},
		},
		{
			name:         "wildcard invalid sni",
			internalName: "internal.host.com",
			apiserver: &configv1.APIServer{
				Spec: configv1.APIServerSpec{
					ServingCerts: configv1.APIServerServingCerts{
						NamedCertificates: []configv1.APIServerNamedServingCert{
							{Names: []string{"internal.*"}},
						},
					},
				},
			},
			validateErrors: func(t *testing.T, errs field.ErrorList) {
				t.Helper()
				if len(errs) != 1 {
					t.Fatal(errs)
				}
				if errs[0].Error() != `spec.servingCerts[0].names[0]: Invalid value: "internal.*": may not match internal loadbalancer: "internal.host.com"` {
					t.Error(errs[0])
				}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fakeclient := configclientfake.NewSimpleClientset(&configv1.Infrastructure{
				ObjectMeta: metav1.ObjectMeta{Name: "cluster"},
				Status: configv1.InfrastructureStatus{
					APIServerInternalURL: test.internalName,
				},
			})

			instance := apiserverV1{
				infrastructureGetter: func() configv1client.InfrastructuresGetter {
					return fakeclient.ConfigV1()
				},
			}
			test.validateErrors(t, instance.validateSNINames(test.apiserver))
		})

	}
}
