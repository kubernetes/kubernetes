package podcertificaterequest

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/kubernetes/pkg/apis/certificates"
)

func TestWarningsOnCreate(t *testing.T) {
	var wantWarnings []string
	gotWarnings := Strategy.WarningsOnCreate(context.Background(), &certificates.PodCertificateRequest{})
	if diff := cmp.Diff(gotWarnings, wantWarnings); diff != "" {
		t.Errorf("Got wrong warnings; diff (-got +want):\n%s", diff)
	}
}

func TestAllowCreateOnUpdate(t *testing.T) {
	if Strategy.AllowCreateOnUpdate() != false {
		t.Errorf("Got true, want false")
	}
}

func TestWarningsOnUpdate(t *testing.T) {
	var wantWarnings []string
	gotWarnings := Strategy.WarningsOnUpdate(context.Background(), &certificates.PodCertificateRequest{}, &certificates.PodCertificateRequest{})
	if diff := cmp.Diff(gotWarnings, wantWarnings); diff != "" {
		t.Errorf("Got wrong warnings; diff (-got +want):\n%s", diff)
	}
}

func TestAllowUnconditionalUpdate(t *testing.T) {
	if Strategy.AllowUnconditionalUpdate() != false {
		t.Errorf("Got true, want false")
	}
}
