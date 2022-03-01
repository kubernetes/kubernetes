// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package target

import (
	"fmt"
	"strings"

	"github.com/pkg/errors"
	"sigs.k8s.io/kustomize/api/konfig"
)

type errMissingKustomization struct {
	path string
}

func (e *errMissingKustomization) Error() string {
	return fmt.Sprintf(
		"unable to find one of %v in directory '%s'",
		commaOr(quoted(konfig.RecognizedKustomizationFileNames())),
		e.path)
}

func IsMissingKustomizationFileError(err error) bool {
	_, ok := err.(*errMissingKustomization)
	if ok {
		return true
	}
	_, ok = errors.Cause(err).(*errMissingKustomization)
	return ok
}

func NewErrMissingKustomization(p string) *errMissingKustomization {
	return &errMissingKustomization{path: p}
}

func quoted(l []string) []string {
	r := make([]string, len(l))
	for i, v := range l {
		r[i] = "'" + v + "'"
	}
	return r
}

func commaOr(q []string) string {
	return strings.Join(q[:len(q)-1], ", ") + " or " + q[len(q)-1]
}
