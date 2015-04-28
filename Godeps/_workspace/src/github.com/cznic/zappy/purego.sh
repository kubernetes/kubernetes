# Copyright 2014 The zappy Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e
set -v

CGO_ENABLED=0 go test -purego true
CGO_ENABLED=0 go test -purego true -tags purego 
CGO_ENABLED=1 go test -purego false
CGO_ENABLED=1 go test -purego true -tags purego 
