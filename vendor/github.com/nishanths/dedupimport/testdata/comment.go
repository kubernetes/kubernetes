//dedupimport -keep comment

package pkg

import (
	"code.org/frontend"
	"code.org/frontend" // line comment
)

import (
	fe "code.org/frontend"
)

var x = frontend.Client
