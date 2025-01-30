/*
Copyright 2019 The Kubernetes Authors.

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

package main

import (
	"fmt"
	"go/ast"
	"go/token"
)

const (
	errNotDirectCall                = "Opts for STABLE metric was not directly passed to new metric function"
	errPositionalArguments          = "Positional arguments are not supported"
	errStabilityLevel               = "StabilityLevel should be passed STABLE, BETA, ALPHA or removed"
	errInvalidNewMetricCall         = "Invalid new metric call, please ensure code compiles"
	errNonStringAttribute           = "Non string attribute is not supported"
	errBadVariableAttribute         = "Metric attribute was not correctly set. Please use only global consts in same file"
	errBadImportedVariableAttribute = "Metric attribute was not correctly set. Please use only global consts in correctly imported same file"
	errFieldNotSupported            = "Field %s is not supported"
	errBuckets                      = "Buckets should be set to list of floats, result from function call of prometheus.LinearBuckets or prometheus.ExponentialBuckets"
	errObjectives                   = "Objectives should be set to map of floats to floats"
	errDecodeUint32                 = "Should decode to uint32"
	errDecodeInt64                  = "Should decode to int64"

	errLabels       = "Labels were not set to list of strings"
	errImport       = `Importing using "." is not supported`
	errExprNotIdent = "expr selector does not refer to type ast.Ident, is type %s"

	errorDecodingString            = "can't decode string"
	errorDecodingLabels            = "can't decode labels"
	errorDecodingConstLabels       = "can't decode const labels"
	errorDecodingStabilityLevel    = "can't decode stability level"
	errorFindingVariableForBuckets = "couldn't find variable for bucket"
	errorFindingVariableForLabels  = "couldn't find variable for labels"
)

type decodeError struct {
	msg string
	pos token.Pos
}

func newDecodeErrorf(node ast.Node, format string, a ...interface{}) *decodeError {
	return &decodeError{
		msg: fmt.Sprintf(format, a...),
		pos: node.Pos(),
	}
}

var _ error = (*decodeError)(nil)

func (e decodeError) Error() string {
	return e.msg
}

func (e decodeError) errorWithFileInformation(fileset *token.FileSet) error {
	position := fileset.Position(e.pos)
	return fmt.Errorf("%s:%d:%d: %s", position.Filename, position.Line, position.Column, e.msg)
}
