// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package externalaccount

import "context"

type programmaticRefreshCredentialSource struct {
	supplierOptions      SupplierOptions
	subjectTokenSupplier SubjectTokenSupplier
	ctx                  context.Context
}

func (cs programmaticRefreshCredentialSource) credentialSourceType() string {
	return "programmatic"
}

func (cs programmaticRefreshCredentialSource) subjectToken() (string, error) {
	return cs.subjectTokenSupplier.SubjectToken(cs.ctx, cs.supplierOptions)
}
