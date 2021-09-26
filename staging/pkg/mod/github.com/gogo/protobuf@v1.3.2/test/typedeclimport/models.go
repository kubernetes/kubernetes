package typedeclimport

import subpkg "github.com/gogo/protobuf/test/typedeclimport/subpkg"

type SomeMessage struct {
	Imported subpkg.AnotherMessage
}
