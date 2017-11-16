// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build appenginevm

package google

func init() {
	appengineFlex = true // Flex doesn't support appengine.AccessToken; depend on metadata server.
}
