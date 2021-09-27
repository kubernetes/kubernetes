package pretty

import (
	"fmt"
	"io"
	"reflect"
)

type sbuf []string

func (p *sbuf) Printf(format string, a ...interface{}) {
	s := fmt.Sprintf(format, a...)
	*p = append(*p, s)
}

// Diff returns a slice where each element describes
// a difference between a and b.
func Diff(a, b interface{}) (desc []string) {
	Pdiff((*sbuf)(&desc), a, b)
	return desc
}

// wprintfer calls Fprintf on w for each Printf call
// with a trailing newline.
type wprintfer struct{ w io.Writer }

func (p *wprintfer) Printf(format string, a ...interface{}) {
	fmt.Fprintf(p.w, format+"\n", a...)
}

// Fdiff writes to w a description of the differences between a and b.
func Fdiff(w io.Writer, a, b interface{}) {
	Pdiff(&wprintfer{w}, a, b)
}

type Printfer interface {
	Printf(format string, a ...interface{})
}

// Pdiff prints to p a description of the differences between a and b.
// It calls Printf once for each difference, with no trailing newline.
// The standard library log.Logger is a Printfer.
func Pdiff(p Printfer, a, b interface{}) {
	d := diffPrinter{
		w:        p,
		aVisited: make(map[visit]visit),
		bVisited: make(map[visit]visit),
	}
	d.diff(reflect.ValueOf(a), reflect.ValueOf(b))
}

type Logfer interface {
	Logf(format string, a ...interface{})
}

// logprintfer calls Fprintf on w for each Printf call
// with a trailing newline.
type logprintfer struct{ l Logfer }

func (p *logprintfer) Printf(format string, a ...interface{}) {
	p.l.Logf(format, a...)
}

// Ldiff prints to l a description of the differences between a and b.
// It calls Logf once for each difference, with no trailing newline.
// The standard library testing.T and testing.B are Logfers.
func Ldiff(l Logfer, a, b interface{}) {
	Pdiff(&logprintfer{l}, a, b)
}

type diffPrinter struct {
	w Printfer
	l string // label

	aVisited map[visit]visit
	bVisited map[visit]visit
}

func (w diffPrinter) printf(f string, a ...interface{}) {
	var l string
	if w.l != "" {
		l = w.l + ": "
	}
	w.w.Printf(l+f, a...)
}

func (w diffPrinter) diff(av, bv reflect.Value) {
	if !av.IsValid() && bv.IsValid() {
		w.printf("nil != %# v", formatter{v: bv, quote: true})
		return
	}
	if av.IsValid() && !bv.IsValid() {
		w.printf("%# v != nil", formatter{v: av, quote: true})
		return
	}
	if !av.IsValid() && !bv.IsValid() {
		return
	}

	at := av.Type()
	bt := bv.Type()
	if at != bt {
		w.printf("%v != %v", at, bt)
		return
	}

	if av.CanAddr() && bv.CanAddr() {
		avis := visit{av.UnsafeAddr(), at}
		bvis := visit{bv.UnsafeAddr(), bt}
		var cycle bool

		// Have we seen this value before?
		if vis, ok := w.aVisited[avis]; ok {
			cycle = true
			if vis != bvis {
				w.printf("%# v (previously visited) != %# v", formatter{v: av, quote: true}, formatter{v: bv, quote: true})
			}
		} else if _, ok := w.bVisited[bvis]; ok {
			cycle = true
			w.printf("%# v != %# v (previously visited)", formatter{v: av, quote: true}, formatter{v: bv, quote: true})
		}
		w.aVisited[avis] = bvis
		w.bVisited[bvis] = avis
		if cycle {
			return
		}
	}

	switch kind := at.Kind(); kind {
	case reflect.Bool:
		if a, b := av.Bool(), bv.Bool(); a != b {
			w.printf("%v != %v", a, b)
		}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		if a, b := av.Int(), bv.Int(); a != b {
			w.printf("%d != %d", a, b)
		}
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		if a, b := av.Uint(), bv.Uint(); a != b {
			w.printf("%d != %d", a, b)
		}
	case reflect.Float32, reflect.Float64:
		if a, b := av.Float(), bv.Float(); a != b {
			w.printf("%v != %v", a, b)
		}
	case reflect.Complex64, reflect.Complex128:
		if a, b := av.Complex(), bv.Complex(); a != b {
			w.printf("%v != %v", a, b)
		}
	case reflect.Array:
		n := av.Len()
		for i := 0; i < n; i++ {
			w.relabel(fmt.Sprintf("[%d]", i)).diff(av.Index(i), bv.Index(i))
		}
	case reflect.Chan, reflect.Func, reflect.UnsafePointer:
		if a, b := av.Pointer(), bv.Pointer(); a != b {
			w.printf("%#x != %#x", a, b)
		}
	case reflect.Interface:
		w.diff(av.Elem(), bv.Elem())
	case reflect.Map:
		ak, both, bk := keyDiff(av.MapKeys(), bv.MapKeys())
		for _, k := range ak {
			w := w.relabel(fmt.Sprintf("[%#v]", k))
			w.printf("%q != (missing)", av.MapIndex(k))
		}
		for _, k := range both {
			w := w.relabel(fmt.Sprintf("[%#v]", k))
			w.diff(av.MapIndex(k), bv.MapIndex(k))
		}
		for _, k := range bk {
			w := w.relabel(fmt.Sprintf("[%#v]", k))
			w.printf("(missing) != %q", bv.MapIndex(k))
		}
	case reflect.Ptr:
		switch {
		case av.IsNil() && !bv.IsNil():
			w.printf("nil != %# v", formatter{v: bv, quote: true})
		case !av.IsNil() && bv.IsNil():
			w.printf("%# v != nil", formatter{v: av, quote: true})
		case !av.IsNil() && !bv.IsNil():
			w.diff(av.Elem(), bv.Elem())
		}
	case reflect.Slice:
		lenA := av.Len()
		lenB := bv.Len()
		if lenA != lenB {
			w.printf("%s[%d] != %s[%d]", av.Type(), lenA, bv.Type(), lenB)
			break
		}
		for i := 0; i < lenA; i++ {
			w.relabel(fmt.Sprintf("[%d]", i)).diff(av.Index(i), bv.Index(i))
		}
	case reflect.String:
		if a, b := av.String(), bv.String(); a != b {
			w.printf("%q != %q", a, b)
		}
	case reflect.Struct:
		for i := 0; i < av.NumField(); i++ {
			w.relabel(at.Field(i).Name).diff(av.Field(i), bv.Field(i))
		}
	default:
		panic("unknown reflect Kind: " + kind.String())
	}
}

func (d diffPrinter) relabel(name string) (d1 diffPrinter) {
	d1 = d
	if d.l != "" && name[0] != '[' {
		d1.l += "."
	}
	d1.l += name
	return d1
}

// keyEqual compares a and b for equality.
// Both a and b must be valid map keys.
func keyEqual(av, bv reflect.Value) bool {
	if !av.IsValid() && !bv.IsValid() {
		return true
	}
	if !av.IsValid() || !bv.IsValid() || av.Type() != bv.Type() {
		return false
	}
	switch kind := av.Kind(); kind {
	case reflect.Bool:
		a, b := av.Bool(), bv.Bool()
		return a == b
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		a, b := av.Int(), bv.Int()
		return a == b
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		a, b := av.Uint(), bv.Uint()
		return a == b
	case reflect.Float32, reflect.Float64:
		a, b := av.Float(), bv.Float()
		return a == b
	case reflect.Complex64, reflect.Complex128:
		a, b := av.Complex(), bv.Complex()
		return a == b
	case reflect.Array:
		for i := 0; i < av.Len(); i++ {
			if !keyEqual(av.Index(i), bv.Index(i)) {
				return false
			}
		}
		return true
	case reflect.Chan, reflect.UnsafePointer, reflect.Ptr:
		a, b := av.Pointer(), bv.Pointer()
		return a == b
	case reflect.Interface:
		return keyEqual(av.Elem(), bv.Elem())
	case reflect.String:
		a, b := av.String(), bv.String()
		return a == b
	case reflect.Struct:
		for i := 0; i < av.NumField(); i++ {
			if !keyEqual(av.Field(i), bv.Field(i)) {
				return false
			}
		}
		return true
	default:
		panic("invalid map key type " + av.Type().String())
	}
}

func keyDiff(a, b []reflect.Value) (ak, both, bk []reflect.Value) {
	for _, av := range a {
		inBoth := false
		for _, bv := range b {
			if keyEqual(av, bv) {
				inBoth = true
				both = append(both, av)
				break
			}
		}
		if !inBoth {
			ak = append(ak, av)
		}
	}
	for _, bv := range b {
		inBoth := false
		for _, av := range a {
			if keyEqual(av, bv) {
				inBoth = true
				break
			}
		}
		if !inBoth {
			bk = append(bk, bv)
		}
	}
	return
}
