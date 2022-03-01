// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

// Package starlark contains a kio.Filter which can be applied to resources to transform
// them through starlark program.
//
// Starlark has become a popular runtime embedding in go programs, especially for Kubernetes
// and data processing.
// Examples: https://github.com/cruise-automation/isopod, https://qri.io/docs/starlark/starlib,
// https://github.com/stripe/skycfg, https://github.com/k14s/ytt
//
// The resources are provided to the starlark program through the global variable "resourceList".
// "resourceList" is a dictionary containing an "items" field with a list of resources.
// The starlark modified "resourceList" is the Filter output.
//
// After being run through the starlark program, the filter will copy the comments from the input
// resources to restore them -- due to them being dropped as a result of serializing the resources
// as starlark values.
//
// "resourceList" may also contain a "functionConfig" entry to configure the starlark script itself.
// Changes made by the starlark program to the "functionConfig" will be reflected in the
// Filter.FunctionConfig value.
//
// The Filter will also format the output so that output has the preferred field ordering
// rather than an alphabetical field ordering.
//
// The resourceList variable adheres to the kustomize function spec as specified by:
// https://github.com/kubernetes-sigs/kustomize/blob/master/cmd/config/docs/api-conventions/functions-spec.md
//
// All items in the resourceList are resources represented as starlark dictionaries/
// The items in the resourceList respect the io spec specified by:
// https://github.com/kubernetes-sigs/kustomize/blob/master/cmd/config/docs/api-conventions/config-io.md
//
// The starlark language spec can be found here:
// https://github.com/google/starlark-go/blob/master/doc/spec.md
package starlark
