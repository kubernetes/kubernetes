package hcl

import (
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strconv"
	"strings"

	"github.com/hashicorp/hcl/hcl/ast"
	"github.com/hashicorp/hcl/hcl/parser"
	"github.com/hashicorp/hcl/hcl/token"
)

// This is the tag to use with structures to have settings for HCL
const tagName = "hcl"

var (
	// nodeType holds a reference to the type of ast.Node
	nodeType reflect.Type = findNodeType()
)

// Unmarshal accepts a byte slice as input and writes the
// data to the value pointed to by v.
func Unmarshal(bs []byte, v interface{}) error {
	root, err := parse(bs)
	if err != nil {
		return err
	}

	return DecodeObject(v, root)
}

// Decode reads the given input and decodes it into the structure
// given by `out`.
func Decode(out interface{}, in string) error {
	obj, err := Parse(in)
	if err != nil {
		return err
	}

	return DecodeObject(out, obj)
}

// DecodeObject is a lower-level version of Decode. It decodes a
// raw Object into the given output.
func DecodeObject(out interface{}, n ast.Node) error {
	val := reflect.ValueOf(out)
	if val.Kind() != reflect.Ptr {
		return errors.New("result must be a pointer")
	}

	// If we have the file, we really decode the root node
	if f, ok := n.(*ast.File); ok {
		n = f.Node
	}

	var d decoder
	return d.decode("root", n, val.Elem())
}

type decoder struct {
	stack []reflect.Kind
}

func (d *decoder) decode(name string, node ast.Node, result reflect.Value) error {
	k := result

	// If we have an interface with a valid value, we use that
	// for the check.
	if result.Kind() == reflect.Interface {
		elem := result.Elem()
		if elem.IsValid() {
			k = elem
		}
	}

	// Push current onto stack unless it is an interface.
	if k.Kind() != reflect.Interface {
		d.stack = append(d.stack, k.Kind())

		// Schedule a pop
		defer func() {
			d.stack = d.stack[:len(d.stack)-1]
		}()
	}

	switch k.Kind() {
	case reflect.Bool:
		return d.decodeBool(name, node, result)
	case reflect.Float32, reflect.Float64:
		return d.decodeFloat(name, node, result)
	case reflect.Int, reflect.Int32, reflect.Int64:
		return d.decodeInt(name, node, result)
	case reflect.Interface:
		// When we see an interface, we make our own thing
		return d.decodeInterface(name, node, result)
	case reflect.Map:
		return d.decodeMap(name, node, result)
	case reflect.Ptr:
		return d.decodePtr(name, node, result)
	case reflect.Slice:
		return d.decodeSlice(name, node, result)
	case reflect.String:
		return d.decodeString(name, node, result)
	case reflect.Struct:
		return d.decodeStruct(name, node, result)
	default:
		return &parser.PosError{
			Pos: node.Pos(),
			Err: fmt.Errorf("%s: unknown kind to decode into: %s", name, k.Kind()),
		}
	}
}

func (d *decoder) decodeBool(name string, node ast.Node, result reflect.Value) error {
	switch n := node.(type) {
	case *ast.LiteralType:
		if n.Token.Type == token.BOOL {
			v, err := strconv.ParseBool(n.Token.Text)
			if err != nil {
				return err
			}

			result.Set(reflect.ValueOf(v))
			return nil
		}
	}

	return &parser.PosError{
		Pos: node.Pos(),
		Err: fmt.Errorf("%s: unknown type %T", name, node),
	}
}

func (d *decoder) decodeFloat(name string, node ast.Node, result reflect.Value) error {
	switch n := node.(type) {
	case *ast.LiteralType:
		if n.Token.Type == token.FLOAT || n.Token.Type == token.NUMBER {
			v, err := strconv.ParseFloat(n.Token.Text, 64)
			if err != nil {
				return err
			}

			result.Set(reflect.ValueOf(v).Convert(result.Type()))
			return nil
		}
	}

	return &parser.PosError{
		Pos: node.Pos(),
		Err: fmt.Errorf("%s: unknown type %T", name, node),
	}
}

func (d *decoder) decodeInt(name string, node ast.Node, result reflect.Value) error {
	switch n := node.(type) {
	case *ast.LiteralType:
		switch n.Token.Type {
		case token.NUMBER:
			v, err := strconv.ParseInt(n.Token.Text, 0, 0)
			if err != nil {
				return err
			}

			if result.Kind() == reflect.Interface {
				result.Set(reflect.ValueOf(int(v)))
			} else {
				result.SetInt(v)
			}
			return nil
		case token.STRING:
			v, err := strconv.ParseInt(n.Token.Value().(string), 0, 0)
			if err != nil {
				return err
			}

			if result.Kind() == reflect.Interface {
				result.Set(reflect.ValueOf(int(v)))
			} else {
				result.SetInt(v)
			}
			return nil
		}
	}

	return &parser.PosError{
		Pos: node.Pos(),
		Err: fmt.Errorf("%s: unknown type %T", name, node),
	}
}

func (d *decoder) decodeInterface(name string, node ast.Node, result reflect.Value) error {
	// When we see an ast.Node, we retain the value to enable deferred decoding.
	// Very useful in situations where we want to preserve ast.Node information
	// like Pos
	if result.Type() == nodeType && result.CanSet() {
		result.Set(reflect.ValueOf(node))
		return nil
	}

	var set reflect.Value
	redecode := true

	// For testing types, ObjectType should just be treated as a list. We
	// set this to a temporary var because we want to pass in the real node.
	testNode := node
	if ot, ok := node.(*ast.ObjectType); ok {
		testNode = ot.List
	}

	switch n := testNode.(type) {
	case *ast.ObjectList:
		// If we're at the root or we're directly within a slice, then we
		// decode objects into map[string]interface{}, otherwise we decode
		// them into lists.
		if len(d.stack) == 0 || d.stack[len(d.stack)-1] == reflect.Slice {
			var temp map[string]interface{}
			tempVal := reflect.ValueOf(temp)
			result := reflect.MakeMap(
				reflect.MapOf(
					reflect.TypeOf(""),
					tempVal.Type().Elem()))

			set = result
		} else {
			var temp []map[string]interface{}
			tempVal := reflect.ValueOf(temp)
			result := reflect.MakeSlice(
				reflect.SliceOf(tempVal.Type().Elem()), 0, len(n.Items))
			set = result
		}
	case *ast.ObjectType:
		// If we're at the root or we're directly within a slice, then we
		// decode objects into map[string]interface{}, otherwise we decode
		// them into lists.
		if len(d.stack) == 0 || d.stack[len(d.stack)-1] == reflect.Slice {
			var temp map[string]interface{}
			tempVal := reflect.ValueOf(temp)
			result := reflect.MakeMap(
				reflect.MapOf(
					reflect.TypeOf(""),
					tempVal.Type().Elem()))

			set = result
		} else {
			var temp []map[string]interface{}
			tempVal := reflect.ValueOf(temp)
			result := reflect.MakeSlice(
				reflect.SliceOf(tempVal.Type().Elem()), 0, 1)
			set = result
		}
	case *ast.ListType:
		var temp []interface{}
		tempVal := reflect.ValueOf(temp)
		result := reflect.MakeSlice(
			reflect.SliceOf(tempVal.Type().Elem()), 0, 0)
		set = result
	case *ast.LiteralType:
		switch n.Token.Type {
		case token.BOOL:
			var result bool
			set = reflect.Indirect(reflect.New(reflect.TypeOf(result)))
		case token.FLOAT:
			var result float64
			set = reflect.Indirect(reflect.New(reflect.TypeOf(result)))
		case token.NUMBER:
			var result int
			set = reflect.Indirect(reflect.New(reflect.TypeOf(result)))
		case token.STRING, token.HEREDOC:
			set = reflect.Indirect(reflect.New(reflect.TypeOf("")))
		default:
			return &parser.PosError{
				Pos: node.Pos(),
				Err: fmt.Errorf("%s: cannot decode into interface: %T", name, node),
			}
		}
	default:
		return fmt.Errorf(
			"%s: cannot decode into interface: %T",
			name, node)
	}

	// Set the result to what its supposed to be, then reset
	// result so we don't reflect into this method anymore.
	result.Set(set)

	if redecode {
		// Revisit the node so that we can use the newly instantiated
		// thing and populate it.
		if err := d.decode(name, node, result); err != nil {
			return err
		}
	}

	return nil
}

func (d *decoder) decodeMap(name string, node ast.Node, result reflect.Value) error {
	if item, ok := node.(*ast.ObjectItem); ok {
		node = &ast.ObjectList{Items: []*ast.ObjectItem{item}}
	}

	if ot, ok := node.(*ast.ObjectType); ok {
		node = ot.List
	}

	n, ok := node.(*ast.ObjectList)
	if !ok {
		return &parser.PosError{
			Pos: node.Pos(),
			Err: fmt.Errorf("%s: not an object type for map (%T)", name, node),
		}
	}

	// If we have an interface, then we can address the interface,
	// but not the slice itself, so get the element but set the interface
	set := result
	if result.Kind() == reflect.Interface {
		result = result.Elem()
	}

	resultType := result.Type()
	resultElemType := resultType.Elem()
	resultKeyType := resultType.Key()
	if resultKeyType.Kind() != reflect.String {
		return &parser.PosError{
			Pos: node.Pos(),
			Err: fmt.Errorf("%s: map must have string keys", name),
		}
	}

	// Make a map if it is nil
	resultMap := result
	if result.IsNil() {
		resultMap = reflect.MakeMap(
			reflect.MapOf(resultKeyType, resultElemType))
	}

	// Go through each element and decode it.
	done := make(map[string]struct{})
	for _, item := range n.Items {
		if item.Val == nil {
			continue
		}

		// github.com/hashicorp/terraform/issue/5740
		if len(item.Keys) == 0 {
			return &parser.PosError{
				Pos: node.Pos(),
				Err: fmt.Errorf("%s: map must have string keys", name),
			}
		}

		// Get the key we're dealing with, which is the first item
		keyStr := item.Keys[0].Token.Value().(string)

		// If we've already processed this key, then ignore it
		if _, ok := done[keyStr]; ok {
			continue
		}

		// Determine the value. If we have more than one key, then we
		// get the objectlist of only these keys.
		itemVal := item.Val
		if len(item.Keys) > 1 {
			itemVal = n.Filter(keyStr)
			done[keyStr] = struct{}{}
		}

		// Make the field name
		fieldName := fmt.Sprintf("%s.%s", name, keyStr)

		// Get the key/value as reflection values
		key := reflect.ValueOf(keyStr)
		val := reflect.Indirect(reflect.New(resultElemType))

		// If we have a pre-existing value in the map, use that
		oldVal := resultMap.MapIndex(key)
		if oldVal.IsValid() {
			val.Set(oldVal)
		}

		// Decode!
		if err := d.decode(fieldName, itemVal, val); err != nil {
			return err
		}

		// Set the value on the map
		resultMap.SetMapIndex(key, val)
	}

	// Set the final map if we can
	set.Set(resultMap)
	return nil
}

func (d *decoder) decodePtr(name string, node ast.Node, result reflect.Value) error {
	// Create an element of the concrete (non pointer) type and decode
	// into that. Then set the value of the pointer to this type.
	resultType := result.Type()
	resultElemType := resultType.Elem()
	val := reflect.New(resultElemType)
	if err := d.decode(name, node, reflect.Indirect(val)); err != nil {
		return err
	}

	result.Set(val)
	return nil
}

func (d *decoder) decodeSlice(name string, node ast.Node, result reflect.Value) error {
	// If we have an interface, then we can address the interface,
	// but not the slice itself, so get the element but set the interface
	set := result
	if result.Kind() == reflect.Interface {
		result = result.Elem()
	}
	// Create the slice if it isn't nil
	resultType := result.Type()
	resultElemType := resultType.Elem()
	if result.IsNil() {
		resultSliceType := reflect.SliceOf(resultElemType)
		result = reflect.MakeSlice(
			resultSliceType, 0, 0)
	}

	// Figure out the items we'll be copying into the slice
	var items []ast.Node
	switch n := node.(type) {
	case *ast.ObjectList:
		items = make([]ast.Node, len(n.Items))
		for i, item := range n.Items {
			items[i] = item
		}
	case *ast.ObjectType:
		items = []ast.Node{n}
	case *ast.ListType:
		items = n.List
	default:
		return &parser.PosError{
			Pos: node.Pos(),
			Err: fmt.Errorf("unknown slice type: %T", node),
		}
	}

	for i, item := range items {
		fieldName := fmt.Sprintf("%s[%d]", name, i)

		// Decode
		val := reflect.Indirect(reflect.New(resultElemType))

		// if item is an object that was decoded from ambiguous JSON and
		// flattened, make sure it's expanded if it needs to decode into a
		// defined structure.
		item := expandObject(item, val)

		if err := d.decode(fieldName, item, val); err != nil {
			return err
		}

		// Append it onto the slice
		result = reflect.Append(result, val)
	}

	set.Set(result)
	return nil
}

// expandObject detects if an ambiguous JSON object was flattened to a List which
// should be decoded into a struct, and expands the ast to properly deocode.
func expandObject(node ast.Node, result reflect.Value) ast.Node {
	item, ok := node.(*ast.ObjectItem)
	if !ok {
		return node
	}

	elemType := result.Type()

	// our target type must be a struct
	switch elemType.Kind() {
	case reflect.Ptr:
		switch elemType.Elem().Kind() {
		case reflect.Struct:
			//OK
		default:
			return node
		}
	case reflect.Struct:
		//OK
	default:
		return node
	}

	// A list value will have a key and field name. If it had more fields,
	// it wouldn't have been flattened.
	if len(item.Keys) != 2 {
		return node
	}

	keyToken := item.Keys[0].Token
	item.Keys = item.Keys[1:]

	// we need to un-flatten the ast enough to decode
	newNode := &ast.ObjectItem{
		Keys: []*ast.ObjectKey{
			&ast.ObjectKey{
				Token: keyToken,
			},
		},
		Val: &ast.ObjectType{
			List: &ast.ObjectList{
				Items: []*ast.ObjectItem{item},
			},
		},
	}

	return newNode
}

func (d *decoder) decodeString(name string, node ast.Node, result reflect.Value) error {
	switch n := node.(type) {
	case *ast.LiteralType:
		switch n.Token.Type {
		case token.NUMBER:
			result.Set(reflect.ValueOf(n.Token.Text).Convert(result.Type()))
			return nil
		case token.STRING, token.HEREDOC:
			result.Set(reflect.ValueOf(n.Token.Value()).Convert(result.Type()))
			return nil
		}
	}

	return &parser.PosError{
		Pos: node.Pos(),
		Err: fmt.Errorf("%s: unknown type for string %T", name, node),
	}
}

func (d *decoder) decodeStruct(name string, node ast.Node, result reflect.Value) error {
	var item *ast.ObjectItem
	if it, ok := node.(*ast.ObjectItem); ok {
		item = it
		node = it.Val
	}

	if ot, ok := node.(*ast.ObjectType); ok {
		node = ot.List
	}

	// Handle the special case where the object itself is a literal. Previously
	// the yacc parser would always ensure top-level elements were arrays. The new
	// parser does not make the same guarantees, thus we need to convert any
	// top-level literal elements into a list.
	if _, ok := node.(*ast.LiteralType); ok && item != nil {
		node = &ast.ObjectList{Items: []*ast.ObjectItem{item}}
	}

	list, ok := node.(*ast.ObjectList)
	if !ok {
		return &parser.PosError{
			Pos: node.Pos(),
			Err: fmt.Errorf("%s: not an object type for struct (%T)", name, node),
		}
	}

	// This slice will keep track of all the structs we'll be decoding.
	// There can be more than one struct if there are embedded structs
	// that are squashed.
	structs := make([]reflect.Value, 1, 5)
	structs[0] = result

	// Compile the list of all the fields that we're going to be decoding
	// from all the structs.
	type field struct {
		field reflect.StructField
		val   reflect.Value
	}
	fields := []field{}
	for len(structs) > 0 {
		structVal := structs[0]
		structs = structs[1:]

		structType := structVal.Type()
		for i := 0; i < structType.NumField(); i++ {
			fieldType := structType.Field(i)
			tagParts := strings.Split(fieldType.Tag.Get(tagName), ",")

			// Ignore fields with tag name "-"
			if tagParts[0] == "-" {
				continue
			}

			if fieldType.Anonymous {
				fieldKind := fieldType.Type.Kind()
				if fieldKind != reflect.Struct {
					return &parser.PosError{
						Pos: node.Pos(),
						Err: fmt.Errorf("%s: unsupported type to struct: %s",
							fieldType.Name, fieldKind),
					}
				}

				// We have an embedded field. We "squash" the fields down
				// if specified in the tag.
				squash := false
				for _, tag := range tagParts[1:] {
					if tag == "squash" {
						squash = true
						break
					}
				}

				if squash {
					structs = append(
						structs, result.FieldByName(fieldType.Name))
					continue
				}
			}

			// Normal struct field, store it away
			fields = append(fields, field{fieldType, structVal.Field(i)})
		}
	}

	usedKeys := make(map[string]struct{})
	decodedFields := make([]string, 0, len(fields))
	decodedFieldsVal := make([]reflect.Value, 0)
	unusedKeysVal := make([]reflect.Value, 0)
	for _, f := range fields {
		field, fieldValue := f.field, f.val
		if !fieldValue.IsValid() {
			// This should never happen
			panic("field is not valid")
		}

		// If we can't set the field, then it is unexported or something,
		// and we just continue onwards.
		if !fieldValue.CanSet() {
			continue
		}

		fieldName := field.Name

		tagValue := field.Tag.Get(tagName)
		tagParts := strings.SplitN(tagValue, ",", 2)
		if len(tagParts) >= 2 {
			switch tagParts[1] {
			case "decodedFields":
				decodedFieldsVal = append(decodedFieldsVal, fieldValue)
				continue
			case "key":
				if item == nil {
					return &parser.PosError{
						Pos: node.Pos(),
						Err: fmt.Errorf("%s: %s asked for 'key', impossible",
							name, fieldName),
					}
				}

				fieldValue.SetString(item.Keys[0].Token.Value().(string))
				continue
			case "unusedKeys":
				unusedKeysVal = append(unusedKeysVal, fieldValue)
				continue
			}
		}

		if tagParts[0] != "" {
			fieldName = tagParts[0]
		}

		// Determine the element we'll use to decode. If it is a single
		// match (only object with the field), then we decode it exactly.
		// If it is a prefix match, then we decode the matches.
		filter := list.Filter(fieldName)

		prefixMatches := filter.Children()
		matches := filter.Elem()
		if len(matches.Items) == 0 && len(prefixMatches.Items) == 0 {
			continue
		}

		// Track the used key
		usedKeys[fieldName] = struct{}{}

		// Create the field name and decode. We range over the elements
		// because we actually want the value.
		fieldName = fmt.Sprintf("%s.%s", name, fieldName)
		if len(prefixMatches.Items) > 0 {
			if err := d.decode(fieldName, prefixMatches, fieldValue); err != nil {
				return err
			}
		}
		for _, match := range matches.Items {
			var decodeNode ast.Node = match.Val
			if ot, ok := decodeNode.(*ast.ObjectType); ok {
				decodeNode = &ast.ObjectList{Items: ot.List.Items}
			}

			if err := d.decode(fieldName, decodeNode, fieldValue); err != nil {
				return err
			}
		}

		decodedFields = append(decodedFields, field.Name)
	}

	if len(decodedFieldsVal) > 0 {
		// Sort it so that it is deterministic
		sort.Strings(decodedFields)

		for _, v := range decodedFieldsVal {
			v.Set(reflect.ValueOf(decodedFields))
		}
	}

	return nil
}

// findNodeType returns the type of ast.Node
func findNodeType() reflect.Type {
	var nodeContainer struct {
		Node ast.Node
	}
	value := reflect.ValueOf(nodeContainer).FieldByName("Node")
	return value.Type()
}
