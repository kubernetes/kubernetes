package podcertificaterequest

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/kubernetes/pkg/apis/certificates"
)

func TestWarningsOnCreate(t *testing.T) {
	strategy := NewStrategy()

	var wantWarnings []string
	gotWarnings := strategy.WarningsOnCreate(context.Background(), &certificates.PodCertificateRequest{})
	if diff := cmp.Diff(gotWarnings, wantWarnings); diff != "" {
		t.Errorf("Got wrong warnings; diff (-got +want):\n%s", diff)
	}
}

func TestAllowCreateOnUpdate(t *testing.T) {
	strategy := NewStrategy()
	if strategy.AllowCreateOnUpdate() != false {
		t.Errorf("Got true, want false")
	}
}

func TestWarningsOnUpdate(t *testing.T) {
	strategy := NewStrategy()
	var wantWarnings []string
	gotWarnings := strategy.WarningsOnUpdate(context.Background(), &certificates.PodCertificateRequest{}, &certificates.PodCertificateRequest{})
	if diff := cmp.Diff(gotWarnings, wantWarnings); diff != "" {
		t.Errorf("Got wrong warnings; diff (-got +want):\n%s", diff)
	}
}

func TestAllowUnconditionalUpdate(t *testing.T) {
	strategy := NewStrategy()
	if strategy.AllowUnconditionalUpdate() != false {
		t.Errorf("Got true, want false")
	}
}
