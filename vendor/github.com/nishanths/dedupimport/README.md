## dedupimport

[![Build Status](https://travis-ci.org/nishanths/dedupimport.svg?branch=master)](https://travis-ci.org/nishanths/dedupimport) [![Godoc](https://godoc.org/github.com/nishanths/dedupimport?status.svg)](http://godoc.org/github.com/nishanths/dedupimport)

Remove duplicate imports that have the same import path but different import
names.

```
go get -u github.com/nishanths/dedupimport
```
See [godoc](https://godoc.org/github.com/nishanths/dedupimport) for flags and usage.

## Example

Given the file

```
package pkg

import (
	"code.org/frontend"
	fe "code.org/frontend"
)

var client frontend.Client
var server fe.Server
```

running dedupimport on it with default options

```
dedupimport file.go
```

will produce

```
package pkg

import (
	"code.org/frontend"
)

var client frontend.Client
var server frontend.Server
```
