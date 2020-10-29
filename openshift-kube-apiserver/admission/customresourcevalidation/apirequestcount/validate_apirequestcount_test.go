package apirequestcount

import (
	"context"
	"testing"

	apiv1 "github.com/openshift/api/apiserver/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestApiRequestCountV1_ValidateCreate(t *testing.T) {
	testCases := []struct {
		name        string
		errExpected bool
	}{
		{"nogood", true},
		{"resource.version", false},
		{"resource.groupnonsense", false},
		{"resource.version.group", false},
		{"resource.version.group.with.dots", false},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			errs := apiRequestCountV1{}.ValidateCreate(context.TODO(), &apiv1.APIRequestCount{ObjectMeta: metav1.ObjectMeta{Name: tc.name}})
			if tc.errExpected != (len(errs) != 0) {
				s := "did not expect "
				if tc.errExpected {
					s = "expected "
				}
				t.Errorf("%serrors, but got %d errors: %v", s, len(errs), errs)
			}
		})
	}

}
