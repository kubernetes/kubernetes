// +build codegen

package api

import "strings"

// ExportableName a name which is exportable as a value or name in Go code
func (a *API) ExportableName(name string) string {
	if name == "" {
		return name
	}

	return strings.ToUpper(name[0:1]) + name[1:]
}
