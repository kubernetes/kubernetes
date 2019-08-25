// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package add

import (
	"testing"

	"sigs.k8s.io/kustomize/pkg/fs"
	"sigs.k8s.io/kustomize/pkg/loader"
	"sigs.k8s.io/kustomize/pkg/types"
	"sigs.k8s.io/kustomize/pkg/validators"
)

func TestNewCmdAddSecretIsNotNil(t *testing.T) {
	fSys := fs.MakeFakeFS()
	ldr := loader.NewFileLoaderAtCwd(validators.MakeFakeValidator(), fSys)
	if newCmdAddSecret(fSys, ldr, nil) == nil {
		t.Fatal("newCmdAddSecret shouldn't be nil")
	}
}

func TestMakeSecretArgs(t *testing.T) {
	secretName := "test-secret-name"

	kustomization := &types.Kustomization{
		NamePrefix: "test-name-prefix",
	}

	secretType := "Opaque"

	if len(kustomization.SecretGenerator) != 0 {
		t.Fatal("Initial kustomization should not have any secrets")
	}
	args := findOrMakeSecretArgs(kustomization, secretName, secretType)

	if args == nil {
		t.Fatalf("args should always be non-nil")
	}

	if len(kustomization.SecretGenerator) != 1 {
		t.Fatalf("Kustomization should have newly created secret")
	}

	if &kustomization.SecretGenerator[len(kustomization.SecretGenerator)-1] != args {
		t.Fatalf("Pointer address for newly inserted secret generator should be same")
	}

	args2 := findOrMakeSecretArgs(kustomization, secretName, secretType)

	if args2 != args {
		t.Fatalf("should have returned an existing args with name: %v", secretName)
	}

	if len(kustomization.SecretGenerator) != 1 {
		t.Fatalf("Should not insert secret for an existing name: %v", secretName)
	}
}

func TestMergeFlagsIntoSecretArgs_LiteralSources(t *testing.T) {
	k := &types.Kustomization{}
	args := findOrMakeSecretArgs(k, "foo", "forbidden")
	mergeFlagsIntoGeneratorArgs(
		&args.GeneratorArgs,
		flagsAndArgs{LiteralSources: []string{"k1=v1"}})
	mergeFlagsIntoGeneratorArgs(
		&args.GeneratorArgs,
		flagsAndArgs{LiteralSources: []string{"k2=v2"}})
	if k.SecretGenerator[0].LiteralSources[0] != "k1=v1" {
		t.Fatalf("expected v1")
	}
	if k.SecretGenerator[0].LiteralSources[1] != "k2=v2" {
		t.Fatalf("expected v2")
	}
}

func TestMergeFlagsIntoSecretArgs_FileSources(t *testing.T) {
	k := &types.Kustomization{}
	args := findOrMakeSecretArgs(k, "foo", "forbidden")
	mergeFlagsIntoGeneratorArgs(
		&args.GeneratorArgs,
		flagsAndArgs{FileSources: []string{"file1"}})
	mergeFlagsIntoGeneratorArgs(
		&args.GeneratorArgs,
		flagsAndArgs{FileSources: []string{"file2"}})
	if k.SecretGenerator[0].FileSources[0] != "file1" {
		t.Fatalf("expected file1")
	}
	if k.SecretGenerator[0].FileSources[1] != "file2" {
		t.Fatalf("expected file2")
	}
}

func TestMergeFlagsIntoSecretArgs_EnvSource(t *testing.T) {
	k := &types.Kustomization{}
	args := findOrMakeSecretArgs(k, "foo", "forbidden")
	mergeFlagsIntoGeneratorArgs(
		&args.GeneratorArgs,
		flagsAndArgs{EnvFileSource: "env1"})
	mergeFlagsIntoGeneratorArgs(
		&args.GeneratorArgs,
		flagsAndArgs{EnvFileSource: "env2"})
	if k.SecretGenerator[0].EnvSources[0] != "env1" {
		t.Fatalf("expected env1")
	}
	if k.SecretGenerator[0].EnvSources[1] != "env2" {
		t.Fatalf("expected env2")
	}
}
