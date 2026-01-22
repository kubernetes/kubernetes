// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package message

import (
	"bytes"
	"fmt" // TODO: consider copying interfaces from package fmt to avoid dependency.
	"math"
	"reflect"
	"sync"
	"unicode/utf8"

	"golang.org/x/text/internal/format"
	"golang.org/x/text/internal/number"
	"golang.org/x/text/language"
	"golang.org/x/text/message/catalog"
)

// Strings for use with buffer.WriteString.
// This is less overhead than using buffer.Write with byte arrays.
const (
	commaSpaceString  = ", "
	nilAngleString    = "<nil>"
	nilParenString    = "(nil)"
	nilString         = "nil"
	mapString         = "map["
	percentBangString = "%!"
	missingString     = "(MISSING)"
	badIndexString    = "(BADINDEX)"
	panicString       = "(PANIC="
	extraString       = "%!(EXTRA "
	badWidthString    = "%!(BADWIDTH)"
	badPrecString     = "%!(BADPREC)"
	noVerbString      = "%!(NOVERB)"

	invReflectString = "<invalid reflect.Value>"
)

var printerPool = sync.Pool{
	New: func() interface{} { return new(printer) },
}

// newPrinter allocates a new printer struct or grabs a cached one.
func newPrinter(pp *Printer) *printer {
	p := printerPool.Get().(*printer)
	p.Printer = *pp
	// TODO: cache most of the following call.
	p.catContext = pp.cat.Context(pp.tag, p)

	p.panicking = false
	p.erroring = false
	p.fmt.init(&p.Buffer)
	return p
}

// free saves used printer structs in printerFree; avoids an allocation per invocation.
func (p *printer) free() {
	p.Buffer.Reset()
	p.arg = nil
	p.value = reflect.Value{}
	printerPool.Put(p)
}

// printer is used to store a printer's state.
// It implements "golang.org/x/text/internal/format".State.
type printer struct {
	Printer

	// the context for looking up message translations
	catContext *catalog.Context

	// buffer for accumulating output.
	bytes.Buffer

	// arg holds the current item, as an interface{}.
	arg interface{}
	// value is used instead of arg for reflect values.
	value reflect.Value

	// fmt is used to format basic items such as integers or strings.
	fmt formatInfo

	// panicking is set by catchPanic to avoid infinite panic, recover, panic, ... recursion.
	panicking bool
	// erroring is set when printing an error string to guard against calling handleMethods.
	erroring bool
}

// Language implements "golang.org/x/text/internal/format".State.
func (p *printer) Language() language.Tag { return p.tag }

func (p *printer) Width() (wid int, ok bool) { return p.fmt.Width, p.fmt.WidthPresent }

func (p *printer) Precision() (prec int, ok bool) { return p.fmt.Prec, p.fmt.PrecPresent }

func (p *printer) Flag(b int) bool {
	switch b {
	case '-':
		return p.fmt.Minus
	case '+':
		return p.fmt.Plus || p.fmt.PlusV
	case '#':
		return p.fmt.Sharp || p.fmt.SharpV
	case ' ':
		return p.fmt.Space
	case '0':
		return p.fmt.Zero
	}
	return false
}

// getField gets the i'th field of the struct value.
// If the field is itself is an interface, return a value for
// the thing inside the interface, not the interface itself.
func getField(v reflect.Value, i int) reflect.Value {
	val := v.Field(i)
	if val.Kind() == reflect.Interface && !val.IsNil() {
		val = val.Elem()
	}
	return val
}

func (p *printer) unknownType(v reflect.Value) {
	if !v.IsValid() {
		p.WriteString(nilAngleString)
		return
	}
	p.WriteByte('?')
	p.WriteString(v.Type().String())
	p.WriteByte('?')
}

func (p *printer) badVerb(verb rune) {
	p.erroring = true
	p.WriteString(percentBangString)
	p.WriteRune(verb)
	p.WriteByte('(')
	switch {
	case p.arg != nil:
		p.WriteString(reflect.TypeOf(p.arg).String())
		p.WriteByte('=')
		p.printArg(p.arg, 'v')
	case p.value.IsValid():
		p.WriteString(p.value.Type().String())
		p.WriteByte('=')
		p.printValue(p.value, 'v', 0)
	default:
		p.WriteString(nilAngleString)
	}
	p.WriteByte(')')
	p.erroring = false
}

func (p *printer) fmtBool(v bool, verb rune) {
	switch verb {
	case 't', 'v':
		p.fmt.fmt_boolean(v)
	default:
		p.badVerb(verb)
	}
}

// fmt0x64 formats a uint64 in hexadecimal and prefixes it with 0x or
// not, as requested, by temporarily setting the sharp flag.
func (p *printer) fmt0x64(v uint64, leading0x bool) {
	sharp := p.fmt.Sharp
	p.fmt.Sharp = leading0x
	p.fmt.fmt_integer(v, 16, unsigned, ldigits)
	p.fmt.Sharp = sharp
}

// fmtInteger formats a signed or unsigned integer.
func (p *printer) fmtInteger(v uint64, isSigned bool, verb rune) {
	switch verb {
	case 'v':
		if p.fmt.SharpV && !isSigned {
			p.fmt0x64(v, true)
			return
		}
		fallthrough
	case 'd':
		if p.fmt.Sharp || p.fmt.SharpV {
			p.fmt.fmt_integer(v, 10, isSigned, ldigits)
		} else {
			p.fmtDecimalInt(v, isSigned)
		}
	case 'b':
		p.fmt.fmt_integer(v, 2, isSigned, ldigits)
	case 'o':
		p.fmt.fmt_integer(v, 8, isSigned, ldigits)
	case 'x':
		p.fmt.fmt_integer(v, 16, isSigned, ldigits)
	case 'X':
		p.fmt.fmt_integer(v, 16, isSigned, udigits)
	case 'c':
		p.fmt.fmt_c(v)
	case 'q':
		if v <= utf8.MaxRune {
			p.fmt.fmt_qc(v)
		} else {
			p.badVerb(verb)
		}
	case 'U':
		p.fmt.fmt_unicode(v)
	default:
		p.badVerb(verb)
	}
}

// fmtFloat formats a float. The default precision for each verb
// is specified as last argument in the call to fmt_float.
func (p *printer) fmtFloat(v float64, size int, verb rune) {
	switch verb {
	case 'b':
		p.fmt.fmt_float(v, size, verb, -1)
	case 'v':
		verb = 'g'
		fallthrough
	case 'g', 'G':
		if p.fmt.Sharp || p.fmt.SharpV {
			p.fmt.fmt_float(v, size, verb, -1)
		} else {
			p.fmtVariableFloat(v, size)
		}
	case 'e', 'E':
		if p.fmt.Sharp || p.fmt.SharpV {
			p.fmt.fmt_float(v, size, verb, 6)
		} else {
			p.fmtScientific(v, size, 6)
		}
	case 'f', 'F':
		if p.fmt.Sharp || p.fmt.SharpV {
			p.fmt.fmt_float(v, size, verb, 6)
		} else {
			p.fmtDecimalFloat(v, size, 6)
		}
	default:
		p.badVerb(verb)
	}
}

func (p *printer) setFlags(f *number.Formatter) {
	f.Flags &^= number.ElideSign
	if p.fmt.Plus || p.fmt.Space {
		f.Flags |= number.AlwaysSign
		if !p.fmt.Plus {
			f.Flags |= number.ElideSign
		}
	} else {
		f.Flags &^= number.AlwaysSign
	}
}

func (p *printer) updatePadding(f *number.Formatter) {
	f.Flags &^= number.PadMask
	if p.fmt.Minus {
		f.Flags |= number.PadAfterSuffix
	} else {
		f.Flags |= number.PadBeforePrefix
	}
	f.PadRune = ' '
	f.FormatWidth = uint16(p.fmt.Width)
}

func (p *printer) initDecimal(minFrac, maxFrac int) {
	f := &p.toDecimal
	f.MinIntegerDigits = 1
	f.MaxIntegerDigits = 0
	f.MinFractionDigits = uint8(minFrac)
	f.MaxFractionDigits = int16(maxFrac)
	p.setFlags(f)
	f.PadRune = 0
	if p.fmt.WidthPresent {
		if p.fmt.Zero {
			wid := p.fmt.Width
			// Use significant integers for this.
			// TODO: this is not the same as width, but so be it.
			if f.MinFractionDigits > 0 {
				wid -= 1 + int(f.MinFractionDigits)
			}
			if p.fmt.Plus || p.fmt.Space {
				wid--
			}
			if wid > 0 && wid > int(f.MinIntegerDigits) {
				f.MinIntegerDigits = uint8(wid)
			}
		}
		p.updatePadding(f)
	}
}

func (p *printer) initScientific(minFrac, maxFrac int) {
	f := &p.toScientific
	if maxFrac < 0 {
		f.SetPrecision(maxFrac)
	} else {
		f.SetPrecision(maxFrac + 1)
		f.MinFractionDigits = uint8(minFrac)
		f.MaxFractionDigits = int16(maxFrac)
	}
	f.MinExponentDigits = 2
	p.setFlags(f)
	f.PadRune = 0
	if p.fmt.WidthPresent {
		f.Flags &^= number.PadMask
		if p.fmt.Zero {
			f.PadRune = f.Digit(0)
			f.Flags |= number.PadAfterPrefix
		} else {
			f.PadRune = ' '
			f.Flags |= number.PadBeforePrefix
		}
		p.updatePadding(f)
	}
}

func (p *printer) fmtDecimalInt(v uint64, isSigned bool) {
	var d number.Decimal

	f := &p.toDecimal
	if p.fmt.PrecPresent {
		p.setFlags(f)
		f.MinIntegerDigits = uint8(p.fmt.Prec)
		f.MaxIntegerDigits = 0
		f.MinFractionDigits = 0
		f.MaxFractionDigits = 0
		if p.fmt.WidthPresent {
			p.updatePadding(f)
		}
	} else {
		p.initDecimal(0, 0)
	}
	d.ConvertInt(p.toDecimal.RoundingContext, isSigned, v)

	out := p.toDecimal.Format([]byte(nil), &d)
	p.Buffer.Write(out)
}

func (p *printer) fmtDecimalFloat(v float64, size, prec int) {
	var d number.Decimal
	if p.fmt.PrecPresent {
		prec = p.fmt.Prec
	}
	p.initDecimal(prec, prec)
	d.ConvertFloat(p.toDecimal.RoundingContext, v, size)

	out := p.toDecimal.Format([]byte(nil), &d)
	p.Buffer.Write(out)
}

func (p *printer) fmtVariableFloat(v float64, size int) {
	prec := -1
	if p.fmt.PrecPresent {
		prec = p.fmt.Prec
	}
	var d number.Decimal
	p.initScientific(0, prec)
	d.ConvertFloat(p.toScientific.RoundingContext, v, size)

	// Copy logic of 'g' formatting from strconv. It is simplified a bit as
	// we don't have to mind having prec > len(d.Digits).
	shortest := prec < 0
	ePrec := prec
	if shortest {
		prec = len(d.Digits)
		ePrec = 6
	} else if prec == 0 {
		prec = 1
		ePrec = 1
	}
	exp := int(d.Exp) - 1
	if exp < -4 || exp >= ePrec {
		p.initScientific(0, prec)

		out := p.toScientific.Format([]byte(nil), &d)
		p.Buffer.Write(out)
	} else {
		if prec > int(d.Exp) {
			prec = len(d.Digits)
		}
		if prec -= int(d.Exp); prec < 0 {
			prec = 0
		}
		p.initDecimal(0, prec)

		out := p.toDecimal.Format([]byte(nil), &d)
		p.Buffer.Write(out)
	}
}

func (p *printer) fmtScientific(v float64, size, prec int) {
	var d number.Decimal
	if p.fmt.PrecPresent {
		prec = p.fmt.Prec
	}
	p.initScientific(prec, prec)
	rc := p.toScientific.RoundingContext
	d.ConvertFloat(rc, v, size)

	out := p.toScientific.Format([]byte(nil), &d)
	p.Buffer.Write(out)

}

// fmtComplex formats a complex number v with
// r = real(v) and j = imag(v) as (r+ji) using
// fmtFloat for r and j formatting.
func (p *printer) fmtComplex(v complex128, size int, verb rune) {
	// Make sure any unsupported verbs are found before the
	// calls to fmtFloat to not generate an incorrect error string.
	switch verb {
	case 'v', 'b', 'g', 'G', 'f', 'F', 'e', 'E':
		p.WriteByte('(')
		p.fmtFloat(real(v), size/2, verb)
		// Imaginary part always has a sign.
		if math.IsNaN(imag(v)) {
			// By CLDR's rules, NaNs do not use patterns or signs. As this code
			// relies on AlwaysSign working for imaginary parts, we need to
			// manually handle NaNs.
			f := &p.toScientific
			p.setFlags(f)
			p.updatePadding(f)
			p.setFlags(f)
			nan := f.Symbol(number.SymNan)
			extra := 0
			if w, ok := p.Width(); ok {
				extra = w - utf8.RuneCountInString(nan) - 1
			}
			if f.Flags&number.PadAfterNumber == 0 {
				for ; extra > 0; extra-- {
					p.WriteRune(f.PadRune)
				}
			}
			p.WriteString(f.Symbol(number.SymPlusSign))
			p.WriteString(nan)
			for ; extra > 0; extra-- {
				p.WriteRune(f.PadRune)
			}
			p.WriteString("i)")
			return
		}
		oldPlus := p.fmt.Plus
		p.fmt.Plus = true
		p.fmtFloat(imag(v), size/2, verb)
		p.WriteString("i)") // TODO: use symbol?
		p.fmt.Plus = oldPlus
	default:
		p.badVerb(verb)
	}
}

func (p *printer) fmtString(v string, verb rune) {
	switch verb {
	case 'v':
		if p.fmt.SharpV {
			p.fmt.fmt_q(v)
		} else {
			p.fmt.fmt_s(v)
		}
	case 's':
		p.fmt.fmt_s(v)
	case 'x':
		p.fmt.fmt_sx(v, ldigits)
	case 'X':
		p.fmt.fmt_sx(v, udigits)
	case 'q':
		p.fmt.fmt_q(v)
	case 'm':
		ctx := p.cat.Context(p.tag, rawPrinter{p})
		if ctx.Execute(v) == catalog.ErrNotFound {
			p.WriteString(v)
		}
	default:
		p.badVerb(verb)
	}
}

func (p *printer) fmtBytes(v []byte, verb rune, typeString string) {
	switch verb {
	case 'v', 'd':
		if p.fmt.SharpV {
			p.WriteString(typeString)
			if v == nil {
				p.WriteString(nilParenString)
				return
			}
			p.WriteByte('{')
			for i, c := range v {
				if i > 0 {
					p.WriteString(commaSpaceString)
				}
				p.fmt0x64(uint64(c), true)
			}
			p.WriteByte('}')
		} else {
			p.WriteByte('[')
			for i, c := range v {
				if i > 0 {
					p.WriteByte(' ')
				}
				p.fmt.fmt_integer(uint64(c), 10, unsigned, ldigits)
			}
			p.WriteByte(']')
		}
	case 's':
		p.fmt.fmt_s(string(v))
	case 'x':
		p.fmt.fmt_bx(v, ldigits)
	case 'X':
		p.fmt.fmt_bx(v, udigits)
	case 'q':
		p.fmt.fmt_q(string(v))
	default:
		p.printValue(reflect.ValueOf(v), verb, 0)
	}
}

func (p *printer) fmtPointer(value reflect.Value, verb rune) {
	var u uintptr
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Map, reflect.Ptr, reflect.Slice, reflect.UnsafePointer:
		u = value.Pointer()
	default:
		p.badVerb(verb)
		return
	}

	switch verb {
	case 'v':
		if p.fmt.SharpV {
			p.WriteByte('(')
			p.WriteString(value.Type().String())
			p.WriteString(")(")
			if u == 0 {
				p.WriteString(nilString)
			} else {
				p.fmt0x64(uint64(u), true)
			}
			p.WriteByte(')')
		} else {
			if u == 0 {
				p.fmt.padString(nilAngleString)
			} else {
				p.fmt0x64(uint64(u), !p.fmt.Sharp)
			}
		}
	case 'p':
		p.fmt0x64(uint64(u), !p.fmt.Sharp)
	case 'b', 'o', 'd', 'x', 'X':
		if verb == 'd' {
			p.fmt.Sharp = true // Print as standard go. TODO: does this make sense?
		}
		p.fmtInteger(uint64(u), unsigned, verb)
	default:
		p.badVerb(verb)
	}
}

func (p *printer) catchPanic(arg interface{}, verb rune) {
	if err := recover(); err != nil {
		// If it's a nil pointer, just say "<nil>". The likeliest causes are a
		// Stringer that fails to guard against nil or a nil pointer for a
		// value receiver, and in either case, "<nil>" is a nice result.
		if v := reflect.ValueOf(arg); v.Kind() == reflect.Ptr && v.IsNil() {
			p.WriteString(nilAngleString)
			return
		}
		// Otherwise print a concise panic message. Most of the time the panic
		// value will print itself nicely.
		if p.panicking {
			// Nested panics; the recursion in printArg cannot succeed.
			panic(err)
		}

		oldFlags := p.fmt.Parser
		// For this output we want default behavior.
		p.fmt.ClearFlags()

		p.WriteString(percentBangString)
		p.WriteRune(verb)
		p.WriteString(panicString)
		p.panicking = true
		p.printArg(err, 'v')
		p.panicking = false
		p.WriteByte(')')

		p.fmt.Parser = oldFlags
	}
}

func (p *printer) handleMethods(verb rune) (handled bool) {
	if p.erroring {
		return
	}
	// Is it a Formatter?
	if formatter, ok := p.arg.(format.Formatter); ok {
		handled = true
		defer p.catchPanic(p.arg, verb)
		formatter.Format(p, verb)
		return
	}
	if formatter, ok := p.arg.(fmt.Formatter); ok {
		handled = true
		defer p.catchPanic(p.arg, verb)
		formatter.Format(p, verb)
		return
	}

	// If we're doing Go syntax and the argument knows how to supply it, take care of it now.
	if p.fmt.SharpV {
		if stringer, ok := p.arg.(fmt.GoStringer); ok {
			handled = true
			defer p.catchPanic(p.arg, verb)
			// Print the result of GoString unadorned.
			p.fmt.fmt_s(stringer.GoString())
			return
		}
	} else {
		// If a string is acceptable according to the format, see if
		// the value satisfies one of the string-valued interfaces.
		// Println etc. set verb to %v, which is "stringable".
		switch verb {
		case 'v', 's', 'x', 'X', 'q':
			// Is it an error or Stringer?
			// The duplication in the bodies is necessary:
			// setting handled and deferring catchPanic
			// must happen before calling the method.
			switch v := p.arg.(type) {
			case error:
				handled = true
				defer p.catchPanic(p.arg, verb)
				p.fmtString(v.Error(), verb)
				return

			case fmt.Stringer:
				handled = true
				defer p.catchPanic(p.arg, verb)
				p.fmtString(v.String(), verb)
				return
			}
		}
	}
	return false
}

func (p *printer) printArg(arg interface{}, verb rune) {
	p.arg = arg
	p.value = reflect.Value{}

	if arg == nil {
		switch verb {
		case 'T', 'v':
			p.fmt.padString(nilAngleString)
		default:
			p.badVerb(verb)
		}
		return
	}

	// Special processing considerations.
	// %T (the value's type) and %p (its address) are special; we always do them first.
	switch verb {
	case 'T':
		p.fmt.fmt_s(reflect.TypeOf(arg).String())
		return
	case 'p':
		p.fmtPointer(reflect.ValueOf(arg), 'p')
		return
	}

	// Some types can be done without reflection.
	switch f := arg.(type) {
	case bool:
		p.fmtBool(f, verb)
	case float32:
		p.fmtFloat(float64(f), 32, verb)
	case float64:
		p.fmtFloat(f, 64, verb)
	case complex64:
		p.fmtComplex(complex128(f), 64, verb)
	case complex128:
		p.fmtComplex(f, 128, verb)
	case int:
		p.fmtInteger(uint64(f), signed, verb)
	case int8:
		p.fmtInteger(uint64(f), signed, verb)
	case int16:
		p.fmtInteger(uint64(f), signed, verb)
	case int32:
		p.fmtInteger(uint64(f), signed, verb)
	case int64:
		p.fmtInteger(uint64(f), signed, verb)
	case uint:
		p.fmtInteger(uint64(f), unsigned, verb)
	case uint8:
		p.fmtInteger(uint64(f), unsigned, verb)
	case uint16:
		p.fmtInteger(uint64(f), unsigned, verb)
	case uint32:
		p.fmtInteger(uint64(f), unsigned, verb)
	case uint64:
		p.fmtInteger(f, unsigned, verb)
	case uintptr:
		p.fmtInteger(uint64(f), unsigned, verb)
	case string:
		p.fmtString(f, verb)
	case []byte:
		p.fmtBytes(f, verb, "[]byte")
	case reflect.Value:
		// Handle extractable values with special methods
		// since printValue does not handle them at depth 0.
		if f.IsValid() && f.CanInterface() {
			p.arg = f.Interface()
			if p.handleMethods(verb) {
				return
			}
		}
		p.printValue(f, verb, 0)
	default:
		// If the type is not simple, it might have methods.
		if !p.handleMethods(verb) {
			// Need to use reflection, since the type had no
			// interface methods that could be used for formatting.
			p.printValue(reflect.ValueOf(f), verb, 0)
		}
	}
}

// printValue is similar to printArg but starts with a reflect value, not an interface{} value.
// It does not handle 'p' and 'T' verbs because these should have been already handled by printArg.
func (p *printer) printValue(value reflect.Value, verb rune, depth int) {
	// Handle values with special methods if not already handled by printArg (depth == 0).
	if depth > 0 && value.IsValid() && value.CanInterface() {
		p.arg = value.Interface()
		if p.handleMethods(verb) {
			return
		}
	}
	p.arg = nil
	p.value = value

	switch f := value; value.Kind() {
	case reflect.Invalid:
		if depth == 0 {
			p.WriteString(invReflectString)
		} else {
			switch verb {
			case 'v':
				p.WriteString(nilAngleString)
			default:
				p.badVerb(verb)
			}
		}
	case reflect.Bool:
		p.fmtBool(f.Bool(), verb)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		p.fmtInteger(uint64(f.Int()), signed, verb)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		p.fmtInteger(f.Uint(), unsigned, verb)
	case reflect.Float32:
		p.fmtFloat(f.Float(), 32, verb)
	case reflect.Float64:
		p.fmtFloat(f.Float(), 64, verb)
	case reflect.Complex64:
		p.fmtComplex(f.Complex(), 64, verb)
	case reflect.Complex128:
		p.fmtComplex(f.Complex(), 128, verb)
	case reflect.String:
		p.fmtString(f.String(), verb)
	case reflect.Map:
		if p.fmt.SharpV {
			p.WriteString(f.Type().String())
			if f.IsNil() {
				p.WriteString(nilParenString)
				return
			}
			p.WriteByte('{')
		} else {
			p.WriteString(mapString)
		}
		keys := f.MapKeys()
		for i, key := range keys {
			if i > 0 {
				if p.fmt.SharpV {
					p.WriteString(commaSpaceString)
				} else {
					p.WriteByte(' ')
				}
			}
			p.printValue(key, verb, depth+1)
			p.WriteByte(':')
			p.printValue(f.MapIndex(key), verb, depth+1)
		}
		if p.fmt.SharpV {
			p.WriteByte('}')
		} else {
			p.WriteByte(']')
		}
	case reflect.Struct:
		if p.fmt.SharpV {
			p.WriteString(f.Type().String())
		}
		p.WriteByte('{')
		for i := 0; i < f.NumField(); i++ {
			if i > 0 {
				if p.fmt.SharpV {
					p.WriteString(commaSpaceString)
				} else {
					p.WriteByte(' ')
				}
			}
			if p.fmt.PlusV || p.fmt.SharpV {
				if name := f.Type().Field(i).Name; name != "" {
					p.WriteString(name)
					p.WriteByte(':')
				}
			}
			p.printValue(getField(f, i), verb, depth+1)
		}
		p.WriteByte('}')
	case reflect.Interface:
		value := f.Elem()
		if !value.IsValid() {
			if p.fmt.SharpV {
				p.WriteString(f.Type().String())
				p.WriteString(nilParenString)
			} else {
				p.WriteString(nilAngleString)
			}
		} else {
			p.printValue(value, verb, depth+1)
		}
	case reflect.Array, reflect.Slice:
		switch verb {
		case 's', 'q', 'x', 'X':
			// Handle byte and uint8 slices and arrays special for the above verbs.
			t := f.Type()
			if t.Elem().Kind() == reflect.Uint8 {
				var bytes []byte
				if f.Kind() == reflect.Slice {
					bytes = f.Bytes()
				} else if f.CanAddr() {
					bytes = f.Slice(0, f.Len()).Bytes()
				} else {
					// We have an array, but we cannot Slice() a non-addressable array,
					// so we build a slice by hand. This is a rare case but it would be nice
					// if reflection could help a little more.
					bytes = make([]byte, f.Len())
					for i := range bytes {
						bytes[i] = byte(f.Index(i).Uint())
					}
				}
				p.fmtBytes(bytes, verb, t.String())
				return
			}
		}
		if p.fmt.SharpV {
			p.WriteString(f.Type().String())
			if f.Kind() == reflect.Slice && f.IsNil() {
				p.WriteString(nilParenString)
				return
			}
			p.WriteByte('{')
			for i := 0; i < f.Len(); i++ {
				if i > 0 {
					p.WriteString(commaSpaceString)
				}
				p.printValue(f.Index(i), verb, depth+1)
			}
			p.WriteByte('}')
		} else {
			p.WriteByte('[')
			for i := 0; i < f.Len(); i++ {
				if i > 0 {
					p.WriteByte(' ')
				}
				p.printValue(f.Index(i), verb, depth+1)
			}
			p.WriteByte(']')
		}
	case reflect.Ptr:
		// pointer to array or slice or struct?  ok at top level
		// but not embedded (avoid loops)
		if depth == 0 && f.Pointer() != 0 {
			switch a := f.Elem(); a.Kind() {
			case reflect.Array, reflect.Slice, reflect.Struct, reflect.Map:
				p.WriteByte('&')
				p.printValue(a, verb, depth+1)
				return
			}
		}
		fallthrough
	case reflect.Chan, reflect.Func, reflect.UnsafePointer:
		p.fmtPointer(f, verb)
	default:
		p.unknownType(f)
	}
}

func (p *printer) badArgNum(verb rune) {
	p.WriteString(percentBangString)
	p.WriteRune(verb)
	p.WriteString(badIndexString)
}

func (p *printer) missingArg(verb rune) {
	p.WriteString(percentBangString)
	p.WriteRune(verb)
	p.WriteString(missingString)
}

func (p *printer) doPrintf(fmt string) {
	for p.fmt.Parser.SetFormat(fmt); p.fmt.Scan(); {
		switch p.fmt.Status {
		case format.StatusText:
			p.WriteString(p.fmt.Text())
		case format.StatusSubstitution:
			p.printArg(p.Arg(p.fmt.ArgNum), p.fmt.Verb)
		case format.StatusBadWidthSubstitution:
			p.WriteString(badWidthString)
			p.printArg(p.Arg(p.fmt.ArgNum), p.fmt.Verb)
		case format.StatusBadPrecSubstitution:
			p.WriteString(badPrecString)
			p.printArg(p.Arg(p.fmt.ArgNum), p.fmt.Verb)
		case format.StatusNoVerb:
			p.WriteString(noVerbString)
		case format.StatusBadArgNum:
			p.badArgNum(p.fmt.Verb)
		case format.StatusMissingArg:
			p.missingArg(p.fmt.Verb)
		default:
			panic("unreachable")
		}
	}

	// Check for extra arguments, but only if there was at least one ordered
	// argument. Note that this behavior is necessarily different from fmt:
	// different variants of messages may opt to drop some or all of the
	// arguments.
	if !p.fmt.Reordered && p.fmt.ArgNum < len(p.fmt.Args) && p.fmt.ArgNum != 0 {
		p.fmt.ClearFlags()
		p.WriteString(extraString)
		for i, arg := range p.fmt.Args[p.fmt.ArgNum:] {
			if i > 0 {
				p.WriteString(commaSpaceString)
			}
			if arg == nil {
				p.WriteString(nilAngleString)
			} else {
				p.WriteString(reflect.TypeOf(arg).String())
				p.WriteString("=")
				p.printArg(arg, 'v')
			}
		}
		p.WriteByte(')')
	}
}

func (p *printer) doPrint(a []interface{}) {
	prevString := false
	for argNum, arg := range a {
		isString := arg != nil && reflect.TypeOf(arg).Kind() == reflect.String
		// Add a space between two non-string arguments.
		if argNum > 0 && !isString && !prevString {
			p.WriteByte(' ')
		}
		p.printArg(arg, 'v')
		prevString = isString
	}
}

// doPrintln is like doPrint but always adds a space between arguments
// and a newline after the last argument.
func (p *printer) doPrintln(a []interface{}) {
	for argNum, arg := range a {
		if argNum > 0 {
			p.WriteByte(' ')
		}
		p.printArg(arg, 'v')
	}
	p.WriteByte('\n')
}
