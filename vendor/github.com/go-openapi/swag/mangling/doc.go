// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

// Package mangling provides name mangling capabilities.
//
// Name mangling is an important stage when generating code:
// it helps construct safe program identifiers that abide by the language rules
// and play along with linters.
//
// Examples:
//
// Suppose we get an object name taken from an API spec: "json_object",
//
// We may generate a legit go type name using [NameMangler.ToGoName]: "JsonObject".
//
// We may then locate this type in a source file named using [NameMangler.ToFileName]: "json_object.go".
//
// The methods exposed by the NameMangler are used to generate code in many different contexts, such as:
//
//   - generating exported or unexported go identifiers from a JSON schema or an API spec
//   - generating file names
//   - generating human-readable comments for types and variables
//   - generating JSON-like API identifiers from go code
//   - ...
package mangling
