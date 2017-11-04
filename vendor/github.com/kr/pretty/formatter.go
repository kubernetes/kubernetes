package pretty

import (
	"fmt"
	"io"
	"reflect"
	"strconv"
	"text/tabwriter"

	"github.com/kr/text"
)

const (
	limit = 50
)

type formatter struct {
	x     interface{}
	force bool
	quote bool
}

// Formatter makes a wrapper, f, that will format x as go source with line
// breaks and tabs. Object f responds to the "%v" formatting verb when both the
// "#" and " " (space) flags are set, for example:
//
//     fmt.Sprintf("%# v", Formatter(x))
//
// If one of these two flags is not set, or any other verb is used, f will
// format x according to the usual rules of package fmt.
// In particular, if x satisfies fmt.Formatter, then x.Format will be called.
func Formatter(x interface{}) (f fmt.Formatter) {
	return formatter{x: x, quote: true}
}

func (fo formatter) String() string {
	return fmt.Sprint(fo.x) // unwrap it
}

func (fo formatter) passThrough(f fmt.State, c rune) {
	s := "%"
	for i := 0; i < 128; i++ {
		if f.Flag(i) {
			s += string(i)
		}
	}
	if w, ok := f.Width(); ok {
		s += fmt.Sprintf("%d", w)
	}
	if p, ok := f.Precision(); ok {
		s += fmt.Sprintf(".%d", p)
	}
	s += string(c)
	fmt.Fprintf(f, s, fo.x)
}

func (fo formatter) Format(f fmt.State, c rune) {
	if fo.force || c == 'v' && f.Flag('#') && f.Flag(' ') {
		w := tabwriter.NewWriter(f, 4, 4, 1, ' ', 0)
		p := &printer{tw: w, Writer: w, visited: make(map[visit]int)}
		p.printValue(reflect.ValueOf(fo.x), true, fo.quote)
		w.Flush()
		return
	}
	fo.passThrough(f, c)
}

type printer struct {
	io.Writer
	tw      *tabwriter.Writer
	visited map[visit]int
	depth   int
}

func (p *printer) indent() *printer {
	q := *p
	q.tw = tabwriter.NewWriter(p.Writer, 4, 4, 1, ' ', 0)
	q.Writer = text.NewIndentWriter(q.tw, []byte{'\t'})
	return &q
}

func (p *printer) printInline(v reflect.Value, x interface{}, showType bool) {
	if showType {
		io.WriteString(p, v.Type().String())
		fmt.Fprintf(p, "(%#v)", x)
	} else {
		fmt.Fprintf(p, "%#v", x)
	}
}

// printValue must keep track of already-printed pointer values to avoid
// infinite recursion.
type visit struct {
	v   uintptr
	typ reflect.Type
}

func (p *printer) printValue(v reflect.Value, showType, quote bool) {
	if p.depth > 10 {
		io.WriteString(p, "!%v(DEPTH EXCEEDED)")
		return
	}

	switch v.Kind() {
	case reflect.Bool:
		p.printInline(v, v.Bool(), showType)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		p.printInline(v, v.Int(), showType)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		p.printInline(v, v.Uint(), showType)
	case reflect.Float32, reflect.Float64:
		p.printInline(v, v.Float(), showType)
	case reflect.Complex64, reflect.Complex128:
		fmt.Fprintf(p, "%#v", v.Complex())
	case reflect.String:
		p.fmtString(v.String(), quote)
	case reflect.Map:
		t := v.Type()
		if showType {
			io.WriteString(p, t.String())
		}
		writeByte(p, '{')
		if nonzero(v) {
			expand := !canInline(v.Type())
			pp := p
			if expand {
				writeByte(p, '\n')
				pp = p.indent()
			}
			keys := v.MapKeys()
			for i := 0; i < v.Len(); i++ {
				showTypeInStruct := true
				k := keys[i]
				mv := v.MapIndex(k)
				pp.printValue(k, false, true)
				writeByte(pp, ':')
				if expand {
					writeByte(pp, '\t')
				}
				showTypeInStruct = t.Elem().Kind() == reflect.Interface
				pp.printValue(mv, showTypeInStruct, true)
				if expand {
					io.WriteString(pp, ",\n")
				} else if i < v.Len()-1 {
					io.WriteString(pp, ", ")
				}
			}
			if expand {
				pp.tw.Flush()
			}
		}
		writeByte(p, '}')
	case reflect.Struct:
		t := v.Type()
		if v.CanAddr() {
			addr := v.UnsafeAddr()
			vis := visit{addr, t}
			if vd, ok := p.visited[vis]; ok && vd < p.depth {
				p.fmtString(t.String()+"{(CYCLIC REFERENCE)}", false)
				break // don't print v again
			}
			p.visited[vis] = p.depth
		}

		if showType {
			io.WriteString(p, t.String())
		}
		writeByte(p, '{')
		if nonzero(v) {
			expand := !canInline(v.Type())
			pp := p
			if expand {
				writeByte(p, '\n')
				pp = p.indent()
			}
			for i := 0; i < v.NumField(); i++ {
				showTypeInStruct := true
				if f := t.Field(i); f.Name != "" {
					io.WriteString(pp, f.Name)
					writeByte(pp, ':')
					if expand {
						writeByte(pp, '\t')
					}
					showTypeInStruct = labelType(f.Type)
				}
				pp.printValue(getField(v, i), showTypeInStruct, true)
				if expand {
					io.WriteString(pp, ",\n")
				} else if i < v.NumField()-1 {
					io.WriteString(pp, ", ")
				}
			}
			if expand {
				pp.tw.Flush()
			}
		}
		writeByte(p, '}')
	case reflect.Interface:
		switch e := v.Elem(); {
		case e.Kind() == reflect.Invalid:
			io.WriteString(p, "nil")
		case e.IsValid():
			pp := *p
			pp.depth++
			pp.printValue(e, showType, true)
		default:
			io.WriteString(p, v.Type().String())
			io.WriteString(p, "(nil)")
		}
	case reflect.Array, reflect.Slice:
		t := v.Type()
		if showType {
			io.WriteString(p, t.String())
		}
		if v.Kind() == reflect.Slice && v.IsNil() && showType {
			io.WriteString(p, "(nil)")
			break
		}
		if v.Kind() == reflect.Slice && v.IsNil() {
			io.WriteString(p, "nil")
			break
		}
		writeByte(p, '{')
		expand := !canInline(v.Type())
		pp := p
		if expand {
			writeByte(p, '\n')
			pp = p.indent()
		}
		for i := 0; i < v.Len(); i++ {
			showTypeInSlice := t.Elem().Kind() == reflect.Interface
			pp.printValue(v.Index(i), showTypeInSlice, true)
			if expand {
				io.WriteString(pp, ",\n")
			} else if i < v.Len()-1 {
				io.WriteString(pp, ", ")
			}
		}
		if expand {
			pp.tw.Flush()
		}
		writeByte(p, '}')
	case reflect.Ptr:
		e := v.Elem()
		if !e.IsValid() {
			writeByte(p, '(')
			io.WriteString(p, v.Type().String())
			io.WriteString(p, ")(nil)")
		} else {
			pp := *p
			pp.depth++
			writeByte(pp, '&')
			pp.printValue(e, true, true)
		}
	case reflect.Chan:
		x := v.Pointer()
		if showType {
			writeByte(p, '(')
			io.WriteString(p, v.Type().String())
			fmt.Fprintf(p, ")(%#v)", x)
		} else {
			fmt.Fprintf(p, "%#v", x)
		}
	case reflect.Func:
		io.WriteString(p, v.Type().String())
		io.WriteString(p, " {...}")
	case reflect.UnsafePointer:
		p.printInline(v, v.Pointer(), showType)
	case reflect.Invalid:
		io.WriteString(p, "nil")
	}
}

func canInline(t reflect.Type) bool {
	switch t.Kind() {
	case reflect.Map:
		return !canExpand(t.Elem())
	case reflect.Struct:
		for i := 0; i < t.NumField(); i++ {
			if canExpand(t.Field(i).Type) {
				return false
			}
		}
		return true
	case reflect.Interface:
		return false
	case reflect.Array, reflect.Slice:
		return !canExpand(t.Elem())
	case reflect.Ptr:
		return false
	case reflect.Chan, reflect.Func, reflect.UnsafePointer:
		return false
	}
	return true
}

func canExpand(t reflect.Type) bool {
	switch t.Kind() {
	case reflect.Map, reflect.Struct,
		reflect.Interface, reflect.Array, reflect.Slice,
		reflect.Ptr:
		return true
	}
	return false
}

func labelType(t reflect.Type) bool {
	switch t.Kind() {
	case reflect.Interface, reflect.Struct:
		return true
	}
	return false
}

func (p *printer) fmtString(s string, quote bool) {
	if quote {
		s = strconv.Quote(s)
	}
	io.WriteString(p, s)
}

func tryDeepEqual(a, b interface{}) bool {
	defer func() { recover() }()
	return reflect.DeepEqual(a, b)
}

func writeByte(w io.Writer, b byte) {
	w.Write([]byte{b})
}

func getField(v reflect.Value, i int) reflect.Value {
	val := v.Field(i)
	if val.Kind() == reflect.Interface && !val.IsNil() {
		val = val.Elem()
	}
	return val
}
