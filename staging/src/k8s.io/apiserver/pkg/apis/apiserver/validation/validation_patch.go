/*
Copyright 2026 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package validation

import (
	celgo "github.com/google/cel-go/cel"
	authenticationcel "k8s.io/apiserver/pkg/authentication/cel"
)

// UPSTREAM: <carry>: Export email claim validation functions for use in OpenShift authentication validation

// UsesEmailClaim checks if the given CEL AST references claims.email
func UsesEmailClaim(ast *celgo.Ast) bool {
	return usesEmailClaim(ast)
}

// UsesEmailVerifiedClaim checks if the given CEL AST references claims.email_verified
func UsesEmailVerifiedClaim(ast *celgo.Ast) bool {
	return usesEmailVerifiedClaim(ast)
}

// AnyUsesEmailVerifiedClaim checks if any of the given compilation results reference claims.email_verified
func AnyUsesEmailVerifiedClaim(results []authenticationcel.CompilationResult) bool {
	return anyUsesEmailVerifiedClaim(results)
}
