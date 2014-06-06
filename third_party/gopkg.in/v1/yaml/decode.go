package yaml

import (
	"reflect"
	"strconv"
	"time"
)

const (
	documentNode = 1 << iota
	mappingNode
	sequenceNode
	scalarNode
	aliasNode
)

type node struct {
	kind         int
	line, column int
	tag          string
	value        string
	implicit     bool
	children     []*node
	anchors      map[string]*node
}

// ----------------------------------------------------------------------------
// Parser, produces a node tree out of a libyaml event stream.

type parser struct {
	parser yaml_parser_t
	event  yaml_event_t
	doc    *node
}

func newParser(b []byte) *parser {
	p := parser{}
	if !yaml_parser_initialize(&p.parser) {
		panic("Failed to initialize YAML emitter")
	}

	if len(b) == 0 {
		b = []byte{'\n'}
	}

	yaml_parser_set_input_string(&p.parser, b)

	p.skip()
	if p.event.typ != yaml_STREAM_START_EVENT {
		panic("Expected stream start event, got " + strconv.Itoa(int(p.event.typ)))
	}
	p.skip()
	return &p
}

func (p *parser) destroy() {
	if p.event.typ != yaml_NO_EVENT {
		yaml_event_delete(&p.event)
	}
	yaml_parser_delete(&p.parser)
}

func (p *parser) skip() {
	if p.event.typ != yaml_NO_EVENT {
		if p.event.typ == yaml_STREAM_END_EVENT {
			panic("Attempted to go past the end of stream. Corrupted value?")
		}
		yaml_event_delete(&p.event)
	}
	if !yaml_parser_parse(&p.parser, &p.event) {
		p.fail()
	}
}

func (p *parser) fail() {
	var where string
	var line int
	if p.parser.problem_mark.line != 0 {
		line = p.parser.problem_mark.line
	} else if p.parser.context_mark.line != 0 {
		line = p.parser.context_mark.line
	}
	if line != 0 {
		where = "line " + strconv.Itoa(line) + ": "
	}
	var msg string
	if len(p.parser.problem) > 0 {
		msg = p.parser.problem
	} else {
		msg = "Unknown problem parsing YAML content"
	}
	panic(where + msg)
}

func (p *parser) anchor(n *node, anchor []byte) {
	if anchor != nil {
		p.doc.anchors[string(anchor)] = n
	}
}

func (p *parser) parse() *node {
	switch p.event.typ {
	case yaml_SCALAR_EVENT:
		return p.scalar()
	case yaml_ALIAS_EVENT:
		return p.alias()
	case yaml_MAPPING_START_EVENT:
		return p.mapping()
	case yaml_SEQUENCE_START_EVENT:
		return p.sequence()
	case yaml_DOCUMENT_START_EVENT:
		return p.document()
	case yaml_STREAM_END_EVENT:
		// Happens when attempting to decode an empty buffer.
		return nil
	default:
		panic("Attempted to parse unknown event: " +
			strconv.Itoa(int(p.event.typ)))
	}
	panic("Unreachable")
}

func (p *parser) node(kind int) *node {
	return &node{
		kind:   kind,
		line:   p.event.start_mark.line,
		column: p.event.start_mark.column,
	}
}

func (p *parser) document() *node {
	n := p.node(documentNode)
	n.anchors = make(map[string]*node)
	p.doc = n
	p.skip()
	n.children = append(n.children, p.parse())
	if p.event.typ != yaml_DOCUMENT_END_EVENT {
		panic("Expected end of document event but got " +
			strconv.Itoa(int(p.event.typ)))
	}
	p.skip()
	return n
}

func (p *parser) alias() *node {
	n := p.node(aliasNode)
	n.value = string(p.event.anchor)
	p.skip()
	return n
}

func (p *parser) scalar() *node {
	n := p.node(scalarNode)
	n.value = string(p.event.value)
	n.tag = string(p.event.tag)
	n.implicit = p.event.implicit
	p.anchor(n, p.event.anchor)
	p.skip()
	return n
}

func (p *parser) sequence() *node {
	n := p.node(sequenceNode)
	p.anchor(n, p.event.anchor)
	p.skip()
	for p.event.typ != yaml_SEQUENCE_END_EVENT {
		n.children = append(n.children, p.parse())
	}
	p.skip()
	return n
}

func (p *parser) mapping() *node {
	n := p.node(mappingNode)
	p.anchor(n, p.event.anchor)
	p.skip()
	for p.event.typ != yaml_MAPPING_END_EVENT {
		n.children = append(n.children, p.parse(), p.parse())
	}
	p.skip()
	return n
}

// ----------------------------------------------------------------------------
// Decoder, unmarshals a node into a provided value.

type decoder struct {
	doc     *node
	aliases map[string]bool
}

func newDecoder() *decoder {
	d := &decoder{}
	d.aliases = make(map[string]bool)
	return d
}

// d.setter deals with setters and pointer dereferencing and initialization.
//
// It's a slightly convoluted case to handle properly:
//
// - nil pointers should be initialized, unless being set to nil
// - we don't know at this point yet what's the value to SetYAML() with.
// - we can't separate pointer deref/init and setter checking, because
//   a setter may be found while going down a pointer chain.
//
// Thus, here is how it takes care of it:
//
// - out is provided as a pointer, so that it can be replaced.
// - when looking at a non-setter ptr, *out=ptr.Elem(), unless tag=!!null
// - when a setter is found, *out=interface{}, and a set() function is
//   returned to call SetYAML() with the value of *out once it's defined.
//
func (d *decoder) setter(tag string, out *reflect.Value, good *bool) (set func()) {
	if (*out).Kind() != reflect.Ptr && (*out).CanAddr() {
		setter, _ := (*out).Addr().Interface().(Setter)
		if setter != nil {
			var arg interface{}
			*out = reflect.ValueOf(&arg).Elem()
			return func() {
				*good = setter.SetYAML(tag, arg)
			}
		}
	}
	again := true
	for again {
		again = false
		setter, _ := (*out).Interface().(Setter)
		if tag != "!!null" || setter != nil {
			if pv := (*out); pv.Kind() == reflect.Ptr {
				if pv.IsNil() {
					*out = reflect.New(pv.Type().Elem()).Elem()
					pv.Set((*out).Addr())
				} else {
					*out = pv.Elem()
				}
				setter, _ = pv.Interface().(Setter)
				again = true
			}
		}
		if setter != nil {
			var arg interface{}
			*out = reflect.ValueOf(&arg).Elem()
			return func() {
				*good = setter.SetYAML(tag, arg)
			}
		}
	}
	return nil
}

func (d *decoder) unmarshal(n *node, out reflect.Value) (good bool) {
	switch n.kind {
	case documentNode:
		good = d.document(n, out)
	case scalarNode:
		good = d.scalar(n, out)
	case aliasNode:
		good = d.alias(n, out)
	case mappingNode:
		good = d.mapping(n, out)
	case sequenceNode:
		good = d.sequence(n, out)
	default:
		panic("Internal error: unknown node kind: " + strconv.Itoa(n.kind))
	}
	return
}

func (d *decoder) document(n *node, out reflect.Value) (good bool) {
	if len(n.children) == 1 {
		d.doc = n
		d.unmarshal(n.children[0], out)
		return true
	}
	return false
}

func (d *decoder) alias(n *node, out reflect.Value) (good bool) {
	an, ok := d.doc.anchors[n.value]
	if !ok {
		panic("Unknown anchor '" + n.value + "' referenced")
	}
	if d.aliases[n.value] {
		panic("Anchor '" + n.value + "' value contains itself")
	}
	d.aliases[n.value] = true
	good = d.unmarshal(an, out)
	delete(d.aliases, n.value)
	return good
}

var durationType = reflect.TypeOf(time.Duration(0))

func (d *decoder) scalar(n *node, out reflect.Value) (good bool) {
	var tag string
	var resolved interface{}
	if n.tag == "" && !n.implicit {
		tag = "!!str"
		resolved = n.value
	} else {
		tag, resolved = resolve(n.tag, n.value)
	}
	if set := d.setter(tag, &out, &good); set != nil {
		defer set()
	}
	switch out.Kind() {
	case reflect.String:
		if resolved != nil {
			out.SetString(n.value)
			good = true
		}
	case reflect.Interface:
		if resolved == nil {
			out.Set(reflect.Zero(out.Type()))
		} else {
			out.Set(reflect.ValueOf(resolved))
		}
		good = true
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		switch resolved := resolved.(type) {
		case int:
			if !out.OverflowInt(int64(resolved)) {
				out.SetInt(int64(resolved))
				good = true
			}
		case int64:
			if !out.OverflowInt(resolved) {
				out.SetInt(resolved)
				good = true
			}
		case float64:
			if resolved < 1<<63-1 && !out.OverflowInt(int64(resolved)) {
				out.SetInt(int64(resolved))
				good = true
			}
		case string:
			if out.Type() == durationType {
				d, err := time.ParseDuration(resolved)
				if err == nil {
					out.SetInt(int64(d))
					good = true
				}
			}
		}
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		switch resolved := resolved.(type) {
		case int:
			if resolved >= 0 {
				out.SetUint(uint64(resolved))
				good = true
			}
		case int64:
			if resolved >= 0 {
				out.SetUint(uint64(resolved))
				good = true
			}
		case float64:
			if resolved < 1<<64-1 && !out.OverflowUint(uint64(resolved)) {
				out.SetUint(uint64(resolved))
				good = true
			}
		}
	case reflect.Bool:
		switch resolved := resolved.(type) {
		case bool:
			out.SetBool(resolved)
			good = true
		}
	case reflect.Float32, reflect.Float64:
		switch resolved := resolved.(type) {
		case int:
			out.SetFloat(float64(resolved))
			good = true
		case int64:
			out.SetFloat(float64(resolved))
			good = true
		case float64:
			out.SetFloat(resolved)
			good = true
		}
	case reflect.Ptr:
		switch resolved.(type) {
		case nil:
			out.Set(reflect.Zero(out.Type()))
			good = true
		default:
			if out.Type().Elem() == reflect.TypeOf(resolved) {
				elem := reflect.New(out.Type().Elem())
				elem.Elem().Set(reflect.ValueOf(resolved))
				out.Set(elem)
				good = true
			}
		}
	}
	return good
}

func settableValueOf(i interface{}) reflect.Value {
	v := reflect.ValueOf(i)
	sv := reflect.New(v.Type()).Elem()
	sv.Set(v)
	return sv
}

func (d *decoder) sequence(n *node, out reflect.Value) (good bool) {
	if set := d.setter("!!seq", &out, &good); set != nil {
		defer set()
	}
	var iface reflect.Value
	if out.Kind() == reflect.Interface {
		// No type hints. Will have to use a generic sequence.
		iface = out
		out = settableValueOf(make([]interface{}, 0))
	}

	if out.Kind() != reflect.Slice {
		return false
	}
	et := out.Type().Elem()

	l := len(n.children)
	for i := 0; i < l; i++ {
		e := reflect.New(et).Elem()
		if ok := d.unmarshal(n.children[i], e); ok {
			out.Set(reflect.Append(out, e))
		}
	}
	if iface.IsValid() {
		iface.Set(out)
	}
	return true
}

func (d *decoder) mapping(n *node, out reflect.Value) (good bool) {
	if set := d.setter("!!map", &out, &good); set != nil {
		defer set()
	}
	if out.Kind() == reflect.Struct {
		return d.mappingStruct(n, out)
	}

	if out.Kind() == reflect.Interface {
		// No type hints. Will have to use a generic map.
		iface := out
		out = settableValueOf(make(map[interface{}]interface{}))
		iface.Set(out)
	}

	if out.Kind() != reflect.Map {
		return false
	}
	outt := out.Type()
	kt := outt.Key()
	et := outt.Elem()

	if out.IsNil() {
		out.Set(reflect.MakeMap(outt))
	}
	l := len(n.children)
	for i := 0; i < l; i += 2 {
		if isMerge(n.children[i]) {
			d.merge(n.children[i+1], out)
			continue
		}
		k := reflect.New(kt).Elem()
		if d.unmarshal(n.children[i], k) {
			e := reflect.New(et).Elem()
			if d.unmarshal(n.children[i+1], e) {
				out.SetMapIndex(k, e)
			}
		}
	}
	return true
}

func (d *decoder) mappingStruct(n *node, out reflect.Value) (good bool) {
	sinfo, err := getStructInfo(out.Type())
	if err != nil {
		panic(err)
	}
	name := settableValueOf("")
	l := len(n.children)
	for i := 0; i < l; i += 2 {
		ni := n.children[i]
		if isMerge(ni) {
			d.merge(n.children[i+1], out)
			continue
		}
		if !d.unmarshal(ni, name) {
			continue
		}
		if info, ok := sinfo.FieldsMap[name.String()]; ok {
			var field reflect.Value
			if info.Inline == nil {
				field = out.Field(info.Num)
			} else {
				field = out.FieldByIndex(info.Inline)
			}
			d.unmarshal(n.children[i+1], field)
		}
	}
	return true
}

func (d *decoder) merge(n *node, out reflect.Value) {
	const wantMap = "map merge requires map or sequence of maps as the value"
	switch n.kind {
	case mappingNode:
		d.unmarshal(n, out)
	case aliasNode:
		an, ok := d.doc.anchors[n.value]
		if ok && an.kind != mappingNode {
			panic(wantMap)
		}
		d.unmarshal(n, out)
	case sequenceNode:
		// Step backwards as earlier nodes take precedence.
		for i := len(n.children)-1; i >= 0; i-- {
			ni := n.children[i]
			if ni.kind == aliasNode {
				an, ok := d.doc.anchors[ni.value]
				if ok && an.kind != mappingNode {
					panic(wantMap)
				}
			} else if ni.kind != mappingNode {
				panic(wantMap)
			}
			d.unmarshal(ni, out)
		}
	default:
		panic(wantMap)
	}
}

func isMerge(n *node) bool {
	return n.kind == scalarNode && n.value == "<<" && (n.implicit == true || n.tag == "!!merge" || n.tag == "tag:yaml.org,2002:merge")
}
