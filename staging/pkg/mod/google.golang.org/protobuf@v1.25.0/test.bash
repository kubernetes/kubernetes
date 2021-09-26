#!/bin/bash
# Copyright 2018 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

cd "$(git rev-parse --show-toplevel)"
go test -v -mod=vendor -timeout=60m -count=1 integration_test.go -failfast "$@"
exit $?
