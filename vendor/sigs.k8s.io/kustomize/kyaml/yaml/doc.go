// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

// Package yaml contains libraries for manipulating individual Kubernetes Resource
// Configuration as yaml, keeping yaml structure and comments.
//
// Parsing Resources
//
// Typically Resources will be initialized as collections through the kio package libraries.
// However it is possible to directly initialize Resources using Parse.
//  resource, err := yaml.Parse("apiVersion: apps/v1\nkind: Deployment")
//
// Processing Resources
//
// Individual Resources are manipulated using the Pipe and PipeE to apply Filter functions
// to transform the Resource data.
//  err := resource.PipeE(yaml.SetAnnotation("key", "value"))
//
// If multiple Filter functions are provided to Pipe or PipeE, each function is applied to
// the result of the last function -- e.g. yaml.Lookup(...), yaml.SetField(...)
//
// Field values may also be retrieved using Pipe.
//  annotationValue, err := resource.Pipe(yaml.GetAnnotation("key"))
//
// See http://www.linfo.org/filters.html for a definition of filters.
//
// Common Filters
//
// There are a number of standard filter functions provided by the yaml package.
//
// Working with annotations:
//  [AnnotationSetter{}, AnnotationGetter{}, AnnotationClearer{}]
//
// Working with fields by path:
//  [PathMatcher{}, PathGetter{}]
//
// Working with individual fields on Maps and Objects:
//  [FieldMatcher{}, FieldSetter{}, FieldGetter{}]
//
// Working with individual elements in Sequences:
//  [ElementAppender{}, ElementSetter{}, ElementMatcher{}]
//
// Writing Filters
//
// Users may implement their own filter functions.  When doing so, can be necessary to work with
// the RNode directly rather than through Pipe.  RNode provides a number of functions for doing
// so. See:
//  [GetMeta(), Fields(), Elements(), String()]
package yaml
