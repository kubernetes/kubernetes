// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

import (
	"fmt"
	"strings"

	"sigs.k8s.io/kustomize/kyaml/errors"
)

type errUnableToFind struct {
	// What are we unable to find?
	what string
	// What things did we try?
	attempts []Pair
}

func (e *errUnableToFind) Error() string {
	var m []string
	for _, p := range e.attempts {
		m = append(m, "('"+p.Value+"'; "+p.Key+")")
	}
	return fmt.Sprintf(
		"unable to find %s - tried: %s", e.what, strings.Join(m, ", "))
}

func NewErrUnableToFind(w string, a []Pair) *errUnableToFind {
	return &errUnableToFind{what: w, attempts: a}
}

func IsErrUnableToFind(err error) bool {
	e := &errUnableToFind{}
	return errors.As(err, &e)
}
