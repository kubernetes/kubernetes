package vendor_dep

import "golang.org/x/tools/present"

type VendorsDep interface {
	Foo() present.Elem
}
