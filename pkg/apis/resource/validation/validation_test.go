package validation

import (
    "strings"
    "testing"

    "k8s.io/apimachinery/pkg/util/validation/field"
    "k8s.io/kubernetes/pkg/apis/resource"
)

func TestValidateQualifiedName_RejectsMultipleSlashes(t *testing.T) {
    fldPath := field.NewPath("test")
    name := resource.QualifiedName("a/b/c")
    errs := validateQualifiedName(name, fldPath)
    if len(errs) != 0 {
        t.Fatalf("expected no validation errors for %q (ratcheting enforces only on Create), got: %v", name, errs)
    }
}

func TestValidateFullyQualifiedName_OriginSet(t *testing.T) {
    fldPath := field.NewPath("test")
    name := resource.FullyQualifiedName("prefix/name/extra")
    errs := validateFullyQualifiedName(name, fldPath)
    if len(errs) != 0 {
        t.Fatalf("expected no validation errors for %q (imperative path does not ratchet), got: %v", name, errs)
    }
}
