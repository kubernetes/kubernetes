// +build go1.8

package predeclared

import "go/doc"

func isPredeclaredIdent(name string) bool {
	return doc.IsPredeclared(name)
}
