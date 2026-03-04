package sprig

import (
	"fmt"
	"reflect"
)

// typeIs returns true if the src is the type named in target.
func typeIs(target string, src interface{}) bool {
	return target == typeOf(src)
}

func typeIsLike(target string, src interface{}) bool {
	t := typeOf(src)
	return target == t || "*"+target == t
}

func typeOf(src interface{}) string {
	return fmt.Sprintf("%T", src)
}

func kindIs(target string, src interface{}) bool {
	return target == kindOf(src)
}

func kindOf(src interface{}) string {
	return reflect.ValueOf(src).Kind().String()
}
