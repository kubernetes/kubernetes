// +build !gccgo

package reflect2

import (
	"reflect"
	"sync"
	"unsafe"
)

// typelinks2 for 1.7 ~
//go:linkname typelinks2 reflect.typelinks
func typelinks2() (sections []unsafe.Pointer, offset [][]int32)

// initOnce guards initialization of types and packages
var initOnce sync.Once

var types map[string]reflect.Type
var packages map[string]map[string]reflect.Type

// discoverTypes initializes types and packages
func discoverTypes() {
	types = make(map[string]reflect.Type)
	packages = make(map[string]map[string]reflect.Type)

	loadGoTypes()
}

func loadGoTypes() {
	var obj interface{} = reflect.TypeOf(0)
	sections, offset := typelinks2()
	for i, offs := range offset {
		rodata := sections[i]
		for _, off := range offs {
			(*emptyInterface)(unsafe.Pointer(&obj)).word = resolveTypeOff(unsafe.Pointer(rodata), off)
			typ := obj.(reflect.Type)
			if typ.Kind() == reflect.Ptr && typ.Elem().Kind() == reflect.Struct {
				loadedType := typ.Elem()
				pkgTypes := packages[loadedType.PkgPath()]
				if pkgTypes == nil {
					pkgTypes = map[string]reflect.Type{}
					packages[loadedType.PkgPath()] = pkgTypes
				}
				types[loadedType.String()] = loadedType
				pkgTypes[loadedType.Name()] = loadedType
			}
		}
	}
}

type emptyInterface struct {
	typ  unsafe.Pointer
	word unsafe.Pointer
}

// TypeByName return the type by its name, just like Class.forName in java
func TypeByName(typeName string) Type {
	initOnce.Do(discoverTypes)
	return Type2(types[typeName])
}

// TypeByPackageName return the type by its package and name
func TypeByPackageName(pkgPath string, name string) Type {
	initOnce.Do(discoverTypes)
	pkgTypes := packages[pkgPath]
	if pkgTypes == nil {
		return nil
	}
	return Type2(pkgTypes[name])
}
