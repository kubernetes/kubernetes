package treeprint

import (
	"fmt"
	"reflect"
	"strings"
)

type StructTreeOption int

const (
	StructNameTree StructTreeOption = iota
	StructValueTree
	StructTagTree
	StructTypeTree
	StructTypeSizeTree
)

func FromStruct(v interface{}, opt ...StructTreeOption) (Tree, error) {
	var treeOpt StructTreeOption
	if len(opt) > 0 {
		treeOpt = opt[0]
	}
	switch treeOpt {
	case StructNameTree:
		tree := New()
		err := nameTree(tree, v)
		return tree, err
	case StructValueTree:
		tree := New()
		err := valueTree(tree, v)
		return tree, err
	case StructTagTree:
		tree := New()
		err := tagTree(tree, v)
		return tree, err
	case StructTypeTree:
		tree := New()
		err := typeTree(tree, v)
		return tree, err
	case StructTypeSizeTree:
		tree := New()
		err := typeSizeTree(tree, v)
		return tree, err
	default:
		err := fmt.Errorf("treeprint: invalid StructTreeOption %v", treeOpt)
		return nil, err
	}
}

type FmtFunc func(name string, v interface{}) (string, bool)

func FromStructWithMeta(v interface{}, fmtFunc FmtFunc) (Tree, error) {
	if fmtFunc == nil {
		tree := New()
		err := nameTree(tree, v)
		return tree, err
	}
	tree := New()
	err := metaTree(tree, v, fmtFunc)
	return tree, err
}

func Repr(v interface{}) string {
	tree := New()
	vType := reflect.TypeOf(v)
	vValue := reflect.ValueOf(v)
	_, val, isStruct := getValue(vType, &vValue)
	if !isStruct {
		return fmt.Sprintf("%+v", val.Interface())
	}
	err := valueTree(tree, val.Interface())
	if err != nil {
		return err.Error()
	}
	return tree.String()
}

func nameTree(tree Tree, v interface{}) error {
	typ, val, err := checkType(v)
	if err != nil {
		return err
	}
	fields := typ.NumField()
	for i := 0; i < fields; i++ {
		field := typ.Field(i)
		fieldValue := val.Field(i)
		name, skip, omit := getMeta(field.Name, field.Tag)
		if skip || omit && isEmpty(&fieldValue) {
			continue
		}
		typ, val, isStruct := getValue(field.Type, &fieldValue)
		if !isStruct {
			tree.AddNode(name)
			continue
		} else if subNum := typ.NumField(); subNum == 0 {
			tree.AddNode(name)
			continue
		}
		branch := tree.AddBranch(name)
		if err := nameTree(branch, val.Interface()); err != nil {
			err := fmt.Errorf("%v on struct branch %s", err, name)
			return err
		}
	}
	return nil
}

func getMeta(fieldName string, tag reflect.StructTag) (name string, skip, omit bool) {
	if tagStr := tag.Get("tree"); len(tagStr) > 0 {
		name, omit = tagSpec(tagStr)
	}
	if name == "-" {
		return fieldName, true, omit
	}
	if len(name) == 0 {
		name = fieldName
	} else if trimmed := strings.TrimSpace(name); len(trimmed) == 0 {
		name = fieldName
	}
	return
}

func valueTree(tree Tree, v interface{}) error {
	typ, val, err := checkType(v)
	if err != nil {
		return err
	}
	fields := typ.NumField()
	for i := 0; i < fields; i++ {
		field := typ.Field(i)
		fieldValue := val.Field(i)
		name, skip, omit := getMeta(field.Name, field.Tag)
		if skip || omit && isEmpty(&fieldValue) {
			continue
		}
		typ, val, isStruct := getValue(field.Type, &fieldValue)
		if !isStruct {
			tree.AddMetaNode(val.Interface(), name)
			continue
		} else if subNum := typ.NumField(); subNum == 0 {
			tree.AddMetaNode(val.Interface(), name)
			continue
		}
		branch := tree.AddBranch(name)
		if err := valueTree(branch, val.Interface()); err != nil {
			err := fmt.Errorf("%v on struct branch %s", err, name)
			return err
		}
	}
	return nil
}

func tagTree(tree Tree, v interface{}) error {
	typ, val, err := checkType(v)
	if err != nil {
		return err
	}
	fields := typ.NumField()
	for i := 0; i < fields; i++ {
		field := typ.Field(i)
		fieldValue := val.Field(i)
		name, skip, omit := getMeta(field.Name, field.Tag)
		if skip || omit && isEmpty(&fieldValue) {
			continue
		}
		filteredTag := filterTags(field.Tag)
		typ, val, isStruct := getValue(field.Type, &fieldValue)
		if !isStruct {
			tree.AddMetaNode(filteredTag, name)
			continue
		} else if subNum := typ.NumField(); subNum == 0 {
			tree.AddMetaNode(filteredTag, name)
			continue
		}
		branch := tree.AddMetaBranch(filteredTag, name)
		if err := tagTree(branch, val.Interface()); err != nil {
			err := fmt.Errorf("%v on struct branch %s", err, name)
			return err
		}
	}
	return nil
}

func typeTree(tree Tree, v interface{}) error {
	typ, val, err := checkType(v)
	if err != nil {
		return err
	}
	fields := typ.NumField()
	for i := 0; i < fields; i++ {
		field := typ.Field(i)
		fieldValue := val.Field(i)
		name, skip, omit := getMeta(field.Name, field.Tag)
		if skip || omit && isEmpty(&fieldValue) {
			continue
		}
		typ, val, isStruct := getValue(field.Type, &fieldValue)
		typename := fmt.Sprintf("%T", val.Interface())
		if !isStruct {
			tree.AddMetaNode(typename, name)
			continue
		} else if subNum := typ.NumField(); subNum == 0 {
			tree.AddMetaNode(typename, name)
			continue
		}
		branch := tree.AddMetaBranch(typename, name)
		if err := typeTree(branch, val.Interface()); err != nil {
			err := fmt.Errorf("%v on struct branch %s", err, name)
			return err
		}
	}
	return nil
}

func typeSizeTree(tree Tree, v interface{}) error {
	typ, val, err := checkType(v)
	if err != nil {
		return err
	}
	fields := typ.NumField()
	for i := 0; i < fields; i++ {
		field := typ.Field(i)
		fieldValue := val.Field(i)
		name, skip, omit := getMeta(field.Name, field.Tag)
		if skip || omit && isEmpty(&fieldValue) {
			continue
		}
		typ, val, isStruct := getValue(field.Type, &fieldValue)
		typesize := typ.Size()
		if !isStruct {
			tree.AddMetaNode(typesize, name)
			continue
		} else if subNum := typ.NumField(); subNum == 0 {
			tree.AddMetaNode(typesize, name)
			continue
		}
		branch := tree.AddMetaBranch(typesize, name)
		if err := typeSizeTree(branch, val.Interface()); err != nil {
			err := fmt.Errorf("%v on struct branch %s", err, name)
			return err
		}
	}
	return nil
}

func metaTree(tree Tree, v interface{}, fmtFunc FmtFunc) error {
	typ, val, err := checkType(v)
	if err != nil {
		return err
	}
	fields := typ.NumField()
	for i := 0; i < fields; i++ {
		field := typ.Field(i)
		fieldValue := val.Field(i)
		name, skip, omit := getMeta(field.Name, field.Tag)
		if skip || omit && isEmpty(&fieldValue) {
			continue
		}
		typ, val, isStruct := getValue(field.Type, &fieldValue)
		formatted, show := fmtFunc(name, val.Interface())
		if !isStruct {
			if show {
				tree.AddMetaNode(formatted, name)
				continue
			}
			tree.AddNode(name)
			continue
		} else if subNum := typ.NumField(); subNum == 0 {
			if show {
				tree.AddMetaNode(formatted, name)
				continue
			}
			tree.AddNode(name)
			continue
		}
		var branch Tree
		if show {
			branch = tree.AddMetaBranch(formatted, name)
		} else {
			branch = tree.AddBranch(name)
		}
		if err := metaTree(branch, val.Interface(), fmtFunc); err != nil {
			err := fmt.Errorf("%v on struct branch %s", err, name)
			return err
		}
	}
	return nil
}

func getValue(typ reflect.Type, val *reflect.Value) (reflect.Type, *reflect.Value, bool) {
	switch typ.Kind() {
	case reflect.Ptr:
		typ = typ.Elem()
		if typ.Kind() == reflect.Struct {
			elem := val.Elem()
			return typ, &elem, true
		}
	case reflect.Struct:
		return typ, val, true
	}
	return typ, val, false
}

func checkType(v interface{}) (reflect.Type, *reflect.Value, error) {
	typ := reflect.TypeOf(v)
	val := reflect.ValueOf(v)
	switch typ.Kind() {
	case reflect.Ptr:
		typ = typ.Elem()
		if typ.Kind() != reflect.Struct {
			err := fmt.Errorf("treeprint: %T is not a struct we could work with", v)
			return nil, nil, err
		}
		val = val.Elem()
	case reflect.Struct:
	default:
		err := fmt.Errorf("treeprint: %T is not a struct we could work with", v)
		return nil, nil, err
	}
	return typ, &val, nil
}
