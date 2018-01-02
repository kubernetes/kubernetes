package namespaces

import (
	"context"
	"os"
	"testing"
)

func TestContext(t *testing.T) {
	ctx := context.Background()
	namespace, ok := Namespace(ctx)
	if ok {
		t.Fatal("namespace should not be present")
	}

	if namespace != "" {
		t.Fatalf("namespace should not be defined: got %q", namespace)
	}

	expected := "test"
	nctx := WithNamespace(ctx, expected)

	namespace, ok = Namespace(nctx)
	if !ok {
		t.Fatal("expected to find a namespace")
	}

	if namespace != expected {
		t.Fatalf("unexpected namespace: %q != %q", namespace, expected)
	}
}

func TestNamespaceFromEnv(t *testing.T) {
	oldenv := os.Getenv(NamespaceEnvVar)
	defer os.Setenv(NamespaceEnvVar, oldenv) // restore old env var

	ctx := context.Background()
	namespace, ok := Namespace(ctx)
	if ok {
		t.Fatal("namespace should not be present")
	}

	if namespace != "" {
		t.Fatalf("namespace should not be defined: got %q", namespace)
	}

	expected := "test-namespace"
	os.Setenv(NamespaceEnvVar, expected)
	nctx := NamespaceFromEnv(ctx)

	namespace, ok = Namespace(nctx)
	if !ok {
		t.Fatal("expected to find a namespace")
	}

	if namespace != expected {
		t.Fatalf("unexpected namespace: %q != %q", namespace, expected)
	}
}
