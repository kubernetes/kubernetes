package admission

import (
	"context"
	"testing"

	"k8s.io/kubernetes/pkg/apis/networking"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	kextensions "k8s.io/kubernetes/pkg/apis/extensions"

	"k8s.io/kubernetes/openshift-kube-apiserver/admission/route/apis/ingressadmission"
)

type fakeAuthorizer struct {
	allow authorizer.Decision
	err   error
}

func (a *fakeAuthorizer) Authorize(context.Context, authorizer.Attributes) (authorizer.Decision, string, error) {
	return a.allow, "", a.err
}

func TestAdmission(t *testing.T) {
	var newIngress *networking.Ingress
	var oldIngress *networking.Ingress

	tests := []struct {
		config           *ingressadmission.IngressAdmissionConfig
		testName         string
		oldHost, newHost string
		op               admission.Operation
		admit            bool
		allow            authorizer.Decision
	}{
		{
			admit:    true,
			config:   emptyConfig(),
			op:       admission.Create,
			testName: "No errors on create",
		},
		{
			admit:    true,
			config:   emptyConfig(),
			op:       admission.Update,
			newHost:  "foo.com",
			oldHost:  "foo.com",
			testName: "keeping the host the same should pass",
		},
		{
			admit:    true,
			config:   emptyConfig(),
			op:       admission.Update,
			oldHost:  "foo.com",
			testName: "deleting a hostname should pass",
		},
		{
			admit:    false,
			config:   emptyConfig(),
			op:       admission.Update,
			newHost:  "foo.com",
			oldHost:  "bar.com",
			testName: "changing hostname should fail",
		},
		{
			admit:    true,
			allow:    authorizer.DecisionAllow,
			config:   emptyConfig(),
			op:       admission.Update,
			newHost:  "foo.com",
			oldHost:  "bar.com",
			testName: "changing hostname should succeed if the user has permission",
		},
		{
			admit:    false,
			config:   nil,
			op:       admission.Update,
			newHost:  "foo.com",
			oldHost:  "bar.com",
			testName: "unconfigured plugin should still fail",
		},
		{
			admit:    true,
			config:   testConfigUpdateAllow(),
			op:       admission.Update,
			newHost:  "foo.com",
			oldHost:  "bar.com",
			testName: "Upstream Hostname updates enabled",
		},
		{
			admit:    true,
			config:   testConfigUpdateAllow(),
			op:       admission.Update,
			newHost:  "foo.com",
			testName: "add new hostname with upstream rules",
		},
		{
			admit:    false,
			allow:    authorizer.DecisionNoOpinion,
			config:   emptyConfig(),
			op:       admission.Create,
			newHost:  "foo.com",
			testName: "setting the host should require permission",
		},
		{
			admit:    true,
			allow:    authorizer.DecisionAllow,
			config:   emptyConfig(),
			op:       admission.Create,
			newHost:  "foo.com",
			testName: "setting the host should pass if user has permission",
		},
	}
	for _, test := range tests {
		if len(test.newHost) > 0 {
			newIngress = &networking.Ingress{
				ObjectMeta: metav1.ObjectMeta{Name: "test"},
				Spec: networking.IngressSpec{
					Rules: []networking.IngressRule{
						{
							Host: test.newHost,
						},
					},
				},
			}
		} else {
			//Used to test deleting a hostname
			newIngress = &networking.Ingress{
				ObjectMeta: metav1.ObjectMeta{Name: "test"},
			}
		}
		handler := NewIngressAdmission(test.config)
		handler.SetAuthorizer(&fakeAuthorizer{allow: test.allow})

		if len(test.oldHost) > 0 {
			//Provides the previous state of an ingress object
			oldIngress = &networking.Ingress{
				ObjectMeta: metav1.ObjectMeta{Name: "test"},
				Spec: networking.IngressSpec{
					Rules: []networking.IngressRule{
						{
							Host: test.oldHost,
						},
					},
				},
			}
		} else {
			oldIngress = nil
		}

		err := handler.Validate(context.TODO(), admission.NewAttributesRecord(newIngress, oldIngress, kextensions.Kind("ingresses").WithVersion("Version"), "namespace", newIngress.ObjectMeta.Name, kextensions.Resource("ingresses").WithVersion("version"), "", test.op, nil, false, nil), nil)
		if test.admit && err != nil {
			t.Errorf("%s: expected no error but got: %s", test.testName, err)
		} else if !test.admit && err == nil {
			t.Errorf("%s: expected an error", test.testName)
		}
	}

}

func emptyConfig() *ingressadmission.IngressAdmissionConfig {
	return &ingressadmission.IngressAdmissionConfig{}
}

func testConfigUpdateAllow() *ingressadmission.IngressAdmissionConfig {
	return &ingressadmission.IngressAdmissionConfig{
		AllowHostnameChanges: true,
	}
}
