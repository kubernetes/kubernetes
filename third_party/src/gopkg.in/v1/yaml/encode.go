package yaml

import (
	"reflect"
	"sort"
	"strconv"
	"time"
)

type encoder struct {
	emitter yaml_emitter_t
	event   yaml_event_t
	out     []byte
	flow    bool
}

func newEncoder() (e *encoder) {
	e = &encoder{}
	e.must(yaml_emitter_initialize(&e.emitter))
	yaml_emitter_set_output_string(&e.emitter, &e.out)
	e.must(yaml_stream_start_event_initialize(&e.event, yaml_UTF8_ENCODING))
	e.emit()
	e.must(yaml_document_start_event_initialize(&e.event, nil, nil, true))
	e.emit()
	return e
}

func (e *encoder) finish() {
	e.must(yaml_document_end_event_initialize(&e.event, true))
	e.emit()
	e.emitter.open_ended = false
	e.must(yaml_stream_end_event_initialize(&e.event))
	e.emit()
}

func (e *encoder) destroy() {
	yaml_emitter_delete(&e.emitter)
}

func (e *encoder) emit() {
	// This will internally delete the e.event value.
	if !yaml_emitter_emit(&e.emitter, &e.event) && e.event.typ != yaml_DOCUMENT_END_EVENT && e.event.typ != yaml_STREAM_END_EVENT {
		e.must(false)
	}
}

func (e *encoder) must(ok bool) {
	if !ok {
		msg := e.emitter.problem
		if msg == "" {
			msg = "Unknown problem generating YAML content"
		}
		panic(msg)
	}
}

func (e *encoder) marshal(tag string, in reflect.Value) {
	var value interface{}
	if getter, ok := in.Interface().(Getter); ok {
		tag, value = getter.GetYAML()
		if value == nil {
			e.nilv()
			return
		}
		in = reflect.ValueOf(value)
	}
	switch in.Kind() {
	case reflect.Interface:
		if in.IsNil() {
			e.nilv()
		} else {
			e.marshal(tag, in.Elem())
		}
	case reflect.Map:
		e.mapv(tag, in)
	case reflect.Ptr:
		if in.IsNil() {
			e.nilv()
		} else {
			e.marshal(tag, in.Elem())
		}
	case reflect.Struct:
		e.structv(tag, in)
	case reflect.Slice:
		e.slicev(tag, in)
	case reflect.String:
		e.stringv(tag, in)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		if in.Type() == durationType {
			e.stringv(tag, reflect.ValueOf(in.Interface().(time.Duration).String()))
		} else {
			e.intv(tag, in)
		}
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		e.uintv(tag, in)
	case reflect.Float32, reflect.Float64:
		e.floatv(tag, in)
	case reflect.Bool:
		e.boolv(tag, in)
	default:
		panic("Can't marshal type yet: " + in.Type().String())
	}
}

func (e *encoder) mapv(tag string, in reflect.Value) {
	e.mappingv(tag, func() {
		keys := keyList(in.MapKeys())
		sort.Sort(keys)
		for _, k := range keys {
			e.marshal("", k)
			e.marshal("", in.MapIndex(k))
		}
	})
}

func (e *encoder) structv(tag string, in reflect.Value) {
	sinfo, err := getStructInfo(in.Type())
	if err != nil {
		panic(err)
	}
	e.mappingv(tag, func() {
		for _, info := range sinfo.FieldsList {
			var value reflect.Value
			if info.Inline == nil {
				value = in.Field(info.Num)
			} else {
				value = in.FieldByIndex(info.Inline)
			}
			if info.OmitEmpty && isZero(value) {
				continue
			}
			e.marshal("", reflect.ValueOf(info.Key))
			e.flow = info.Flow
			e.marshal("", value)
		}
	})
}

func (e *encoder) mappingv(tag string, f func()) {
	implicit := tag == ""
	style := yaml_BLOCK_MAPPING_STYLE
	if e.flow {
		e.flow = false
		style = yaml_FLOW_MAPPING_STYLE
	}
	e.must(yaml_mapping_start_event_initialize(&e.event, nil, []byte(tag), implicit, style))
	e.emit()
	f()
	e.must(yaml_mapping_end_event_initialize(&e.event))
	e.emit()
}

func (e *encoder) slicev(tag string, in reflect.Value) {
	implicit := tag == ""
	style := yaml_BLOCK_SEQUENCE_STYLE
	if e.flow {
		e.flow = false
		style = yaml_FLOW_SEQUENCE_STYLE
	}
	e.must(yaml_sequence_start_event_initialize(&e.event, nil, []byte(tag), implicit, style))
	e.emit()
	n := in.Len()
	for i := 0; i < n; i++ {
		e.marshal("", in.Index(i))
	}
	e.must(yaml_sequence_end_event_initialize(&e.event))
	e.emit()
}

func (e *encoder) stringv(tag string, in reflect.Value) {
	var style yaml_scalar_style_t
	s := in.String()
	if rtag, _ := resolve("", s); rtag != "!!str" {
		style = yaml_DOUBLE_QUOTED_SCALAR_STYLE
	} else {
		style = yaml_PLAIN_SCALAR_STYLE
	}
	e.emitScalar(s, "", tag, style)
}

func (e *encoder) boolv(tag string, in reflect.Value) {
	var s string
	if in.Bool() {
		s = "true"
	} else {
		s = "false"
	}
	e.emitScalar(s, "", tag, yaml_PLAIN_SCALAR_STYLE)
}

func (e *encoder) intv(tag string, in reflect.Value) {
	s := strconv.FormatInt(in.Int(), 10)
	e.emitScalar(s, "", tag, yaml_PLAIN_SCALAR_STYLE)
}

func (e *encoder) uintv(tag string, in reflect.Value) {
	s := strconv.FormatUint(in.Uint(), 10)
	e.emitScalar(s, "", tag, yaml_PLAIN_SCALAR_STYLE)
}

func (e *encoder) floatv(tag string, in reflect.Value) {
	// FIXME: Handle 64 bits here.
	s := strconv.FormatFloat(float64(in.Float()), 'g', -1, 32)
	switch s {
	case "+Inf":
		s = ".inf"
	case "-Inf":
		s = "-.inf"
	case "NaN":
		s = ".nan"
	}
	e.emitScalar(s, "", tag, yaml_PLAIN_SCALAR_STYLE)
}

func (e *encoder) nilv() {
	e.emitScalar("null", "", "", yaml_PLAIN_SCALAR_STYLE)
}

func (e *encoder) emitScalar(value, anchor, tag string, style yaml_scalar_style_t) {
	implicit := tag == ""
	if !implicit {
		style = yaml_PLAIN_SCALAR_STYLE
	}
	e.must(yaml_scalar_event_initialize(&e.event, []byte(anchor), []byte(tag), []byte(value), implicit, implicit, style))
	e.emit()
}
