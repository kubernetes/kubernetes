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
    if len(errs) == 0 {
        t.Fatalf("expected validation error for %q, got none", name)
    }
    found := false
    for _, e := range errs {
        if strings.Contains(e.Detail, "must not contain more than one slash") {
            found = true
            break
        }
    }
    if !found {
        t.Fatalf("expected error detail to mention multi-slash, got: %v", errs)
    }
}

func TestValidateFullyQualifiedName_OriginSet(t *testing.T) {
    fldPath := field.NewPath("test")
    name := resource.FullyQualifiedName("prefix/name/extra")
    errs := validateFullyQualifiedName(name, fldPath)
    if len(errs) == 0 {
        t.Fatalf("expected validation error for %q, got none", name)
    }
    for _, e := range errs {
        if e.Origin != "format=k8s-resource-fully-qualified-name" {
            t.Fatalf("expected Origin to be format=k8s-resource-fully-qualified-name, got %q", e.Origin)
        }
    }
}
