// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package message

import (
	"bytes"
	"fmt" // TODO: consider copying interfaces from package fmt to avoid dependency.
	"math"
	"reflect"
	"unicode/utf8"

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

// printer is used to store a printer's state.
// It implements "golang.org/x/text/internal/format".State.
type printer struct {
	// the context for looking up message translations
	catContext *catalog.Context
	// the language
	tag language.Tag

	// buffer for accumulating output.
	bytes.Buffer

	// retain arguments across calls.
	args []interface{}
	// retain current argument number across calls
	argNum int
	// arg holds the current item, as an interface{}.
	arg interface{}
	// value is used instead of arg for reflect values.
	value reflect.Value

	// fmt is used to format basic items such as integers or strings.
	fmt formatInfo

	// reordered records whether the format string used argument reordering.
	reordered bool
	// goodArgNum records whether the most recent reordering directive was valid.
	goodArgNum bool
	// panicking is set by catchPanic to avoid infinite panic, recover, panic, ... recursion.
	panicking bool
	// erroring is set when printing an error string to guard against calling handleMethods.
	erroring bool

	toDecimal    number.Formatter
	toScientific number.Formatter
}

func (p *printer) reset() {
	p.Buffer.Reset()
	p.argNum = 0
	p.reordered = false
	p.panicking = false
	p.erroring = false
	p.fmt.init(&p.Buffer)
}

// Language implements "golang.org/x/text/internal/format".State.
func (p *printer) Language() language.Tag { return p.tag }

func (p *printer) Width() (wid int, ok bool) { return p.fmt.wid, p.fmt.widPresent }

func (p *printer) Precision() (prec int, ok bool) { return p.fmt.prec, p.fmt.precPresent }

func (p *printer) Flag(b int) bool {
	switch b {
	case '-':
		return p.fmt.minus
	case '+':
		return p.fmt.plus || p.fmt.plusV
	case '#':
		return p.fmt.sharp || p.fmt.sharpV
	case ' ':
		return p.fmt.space
	case '0':
		return p.fmt.zero
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

// tooLarge reports whether the magnitude of the integer is
// too large to be used as a formatting width or precision.
func tooLarge(x int) bool {
	const max int = 1e6
	return x > max || x < -max
}

// parsenum converts ASCII to integer.  num is 0 (and isnum is false) if no number present.
func parsenum(s string, start, end int) (num int, isnum bool, newi int) {
	if start >= end {
		return 0, false, end
	}
	for newi = start; newi < end && '0' <= s[newi] && s[newi] <= '9'; newi++ {
		if tooLarge(num) {
			return 0, false, end // Overflow; crazy long number most likely.
		}
		num = num*10 + int(s[newi]-'0')
		isnum = true
	}
	return
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
	sharp := p.fmt.sharp
	p.fmt.sharp = leading0x
	p.fmt.fmt_integer(v, 16, unsigned, ldigits)
	p.fmt.sharp = sharp
}

// fmtInteger formats a signed or unsigned integer.
func (p *printer) fmtInteger(v uint64, isSigned bool, verb rune) {
	switch verb {
	case 'v':
		if p.fmt.sharpV && !isSigned {
			p.fmt0x64(v, true)
			return
		}
		fallthrough
	case 'd':
		if p.fmt.sharp || p.fmt.sharpV {
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
		if p.fmt.sharp || p.fmt.sharpV {
			p.fmt.fmt_float(v, size, verb, -1)
		} else {
			p.fmtVariableFloat(v, size, -1)
		}
	case 'e', 'E':
		if p.fmt.sharp || p.fmt.sharpV {
			p.fmt.fmt_float(v, size, verb, 6)
		} else {
			p.fmtScientific(v, size, 6)
		}
	case 'f', 'F':
		if p.fmt.sharp || p.fmt.sharpV {
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
	if p.fmt.plus || p.fmt.space {
		f.Flags |= number.AlwaysSign
		if !p.fmt.plus {
			f.Flags |= number.ElideSign
		}
	} else {
		f.Flags &^= number.AlwaysSign
	}
}

func (p *printer) updatePadding(f *number.Formatter) {
	f.Flags &^= number.PadMask
	if p.fmt.minus {
		f.Flags |= number.PadAfterSuffix
	} else {
		f.Flags |= number.PadBeforePrefix
	}
	f.PadRune = ' '
	f.FormatWidth = uint16(p.fmt.wid)
}

func (p *printer) initDecimal(minFrac, maxFrac int) {
	f := &p.toDecimal
	f.MinIntegerDigits = 1
	f.MaxIntegerDigits = 0
	f.MinFractionDigits = uint8(minFrac)
	f.MaxFractionDigits = uint8(maxFrac)
	p.setFlags(f)
	f.PadRune = 0
	if p.fmt.widPresent {
		if p.fmt.zero {
			wid := p.fmt.wid
			// Use significant integers for this.
			// TODO: this is not the same as width, but so be it.
			if f.MinFractionDigits > 0 {
				wid -= 1 + int(f.MinFractionDigits)
			}
			if p.fmt.plus || p.fmt.space {
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
	f.MinFractionDigits = uint8(minFrac)
	f.MaxFractionDigits = uint8(maxFrac)
	f.MinExponentDigits = 2
	p.setFlags(f)
	f.PadRune = 0
	if p.fmt.widPresent {
		f.Flags &^= number.PadMask
		if p.fmt.zero {
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
	p.toDecimal.RoundingContext.Scale = 0
	d.ConvertInt(&p.toDecimal.RoundingContext, isSigned, v)

	f := &p.toDecimal
	if p.fmt.precPresent {
		p.setFlags(f)
		f.MinIntegerDigits = uint8(p.fmt.prec)
		f.MaxIntegerDigits = 0
		f.MinFractionDigits = 0
		f.MaxFractionDigits = 0
		if p.fmt.widPresent {
			p.updatePadding(f)
		}
	} else {
		p.initDecimal(0, 0)
	}

	out := p.toDecimal.Format([]byte(nil), &d)
	p.Buffer.Write(out)
}

func (p *printer) fmtDecimalFloat(v float64, size, prec int) {
	var d number.Decimal
	if p.fmt.precPresent {
		prec = p.fmt.prec
	}
	p.toDecimal.RoundingContext.Scale = int32(prec)
	d.ConvertFloat(&p.toDecimal.RoundingContext, v, size)

	p.initDecimal(prec, prec)

	out := p.toDecimal.Format([]byte(nil), &d)
	p.Buffer.Write(out)
}

func (p *printer) fmtVariableFloat(v float64, size, prec int) {
	if p.fmt.precPresent {
		prec = p.fmt.prec
	}
	var d number.Decimal
	p.toScientific.RoundingContext.Precision = int32(prec)
	d.ConvertFloat(&p.toScientific.RoundingContext, v, size)

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
	if p.fmt.precPresent {
		prec = p.fmt.prec
	}
	p.toScientific.RoundingContext.Precision = int32(prec)
	d.ConvertFloat(&p.toScientific.RoundingContext, v, size)

	p.initScientific(prec, prec)

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
		oldPlus := p.fmt.plus
		p.fmt.plus = true
		p.fmtFloat(imag(v), size/2, verb)
		p.WriteString("i)") // TODO: use symbol?
		p.fmt.plus = oldPlus
	default:
		p.badVerb(verb)
	}
}

func (p *printer) fmtString(v string, verb rune) {
	switch verb {
	case 'v':
		if p.fmt.sharpV {
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
	default:
		p.badVerb(verb)
	}
}

func (p *printer) fmtBytes(v []byte, verb rune, typeString string) {
	switch verb {
	case 'v', 'd':
		if p.fmt.sharpV {
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
		if p.fmt.sharpV {
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
				p.fmt0x64(uint64(u), !p.fmt.sharp)
			}
		}
	case 'p':
		p.fmt0x64(uint64(u), !p.fmt.sharp)
	case 'b', 'o', 'd', 'x', 'X':
		if verb == 'd' {
			p.fmt.sharp = true // Print as standard go. TODO: does this make sense?
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

		oldFlags := p.fmt.fmtFlags
		// For this output we want default behavior.
		p.fmt.clearflags()

		p.WriteString(percentBangString)
		p.WriteRune(verb)
		p.WriteString(panicString)
		p.panicking = true
		p.printArg(err, 'v')
		p.panicking = false
		p.WriteByte(')')

		p.fmt.fmtFlags = oldFlags
	}
}

func (p *printer) handleMethods(verb rune) (handled bool) {
	if p.erroring {
		return
	}
	// Is it a Formatter?
	if formatter, ok := p.arg.(fmt.Formatter); ok {
		handled = true
		defer p.catchPanic(p.arg, verb)
		formatter.Format(p, verb)
		return
	}

	// If we're doing Go syntax and the argument knows how to supply it, take care of it now.
	if p.fmt.sharpV {
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
		if p.fmt.sharpV {
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
				if p.fmt.sharpV {
					p.WriteString(commaSpaceString)
				} else {
					p.WriteByte(' ')
				}
			}
			p.printValue(key, verb, depth+1)
			p.WriteByte(':')
			p.printValue(f.MapIndex(key), verb, depth+1)
		}
		if p.fmt.sharpV {
			p.WriteByte('}')
		} else {
			p.WriteByte(']')
		}
	case reflect.Struct:
		if p.fmt.sharpV {
			p.WriteString(f.Type().String())
		}
		p.WriteByte('{')
		for i := 0; i < f.NumField(); i++ {
			if i > 0 {
				if p.fmt.sharpV {
					p.WriteString(commaSpaceString)
				} else {
					p.WriteByte(' ')
				}
			}
			if p.fmt.plusV || p.fmt.sharpV {
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
			if p.fmt.sharpV {
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
		if p.fmt.sharpV {
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

// intFromArg gets the argNumth element of a. On return, isInt reports whether the argument has integer type.
func (p *printer) intFromArg() (num int, isInt bool) {
	if p.argNum < len(p.args) {
		arg := p.args[p.argNum]
		num, isInt = arg.(int) // Almost always OK.
		if !isInt {
			// Work harder.
			switch v := reflect.ValueOf(arg); v.Kind() {
			case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
				n := v.Int()
				if int64(int(n)) == n {
					num = int(n)
					isInt = true
				}
			case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
				n := v.Uint()
				if int64(n) >= 0 && uint64(int(n)) == n {
					num = int(n)
					isInt = true
				}
			default:
				// Already 0, false.
			}
		}
		p.argNum++
		if tooLarge(num) {
			num = 0
			isInt = false
		}
	}
	return
}

// parseArgNumber returns the value of the bracketed number, minus 1
// (explicit argument numbers are one-indexed but we want zero-indexed).
// The opening bracket is known to be present at format[0].
// The returned values are the index, the number of bytes to consume
// up to the closing paren, if present, and whether the number parsed
// ok. The bytes to consume will be 1 if no closing paren is present.
func parseArgNumber(format string) (index int, wid int, ok bool) {
	// There must be at least 3 bytes: [n].
	if len(format) < 3 {
		return 0, 1, false
	}

	// Find closing bracket.
	for i := 1; i < len(format); i++ {
		if format[i] == ']' {
			width, ok, newi := parsenum(format, 1, i)
			if !ok || newi != i {
				return 0, i + 1, false
			}
			return width - 1, i + 1, true // arg numbers are one-indexed and skip paren.
		}
	}
	return 0, 1, false
}

// updateArgNumber returns the next argument to evaluate, which is either the value of the passed-in
// argNum or the value of the bracketed integer that begins format[i:]. It also returns
// the new value of i, that is, the index of the next byte of the format to process.
func (p *printer) updateArgNumber(format string, i int) (newi int, found bool) {
	if len(format) <= i || format[i] != '[' {
		return i, false
	}
	p.reordered = true
	index, wid, ok := parseArgNumber(format[i:])
	if ok && 0 <= index && index < len(p.args) {
		p.argNum = index
		return i + wid, true
	}
	p.goodArgNum = false
	return i + wid, ok
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

func (p *printer) doPrintf(format string) {
	end := len(format)
	afterIndex := false // previous item in format was an index like [3].
formatLoop:
	for i := 0; i < end; {
		p.goodArgNum = true
		lasti := i
		for i < end && format[i] != '%' {
			i++
		}
		if i > lasti {
			p.WriteString(format[lasti:i])
		}
		if i >= end {
			// done processing format string
			break
		}

		// Process one verb
		i++

		// Do we have flags?
		p.fmt.clearflags()
	simpleFormat:
		for ; i < end; i++ {
			c := format[i]
			switch c {
			case '#':
				p.fmt.sharp = true
			case '0':
				p.fmt.zero = !p.fmt.minus // Only allow zero padding to the left.
			case '+':
				p.fmt.plus = true
			case '-':
				p.fmt.minus = true
				p.fmt.zero = false // Do not pad with zeros to the right.
			case ' ':
				p.fmt.space = true
			default:
				// Fast path for common case of ascii lower case simple verbs
				// without precision or width or argument indices.
				if 'a' <= c && c <= 'z' && p.argNum < len(p.args) {
					if c == 'v' {
						// Go syntax
						p.fmt.sharpV = p.fmt.sharp
						p.fmt.sharp = false
						// Struct-field syntax
						p.fmt.plusV = p.fmt.plus
						p.fmt.plus = false
					}
					p.printArg(p.Arg(p.argNum), rune(c))
					p.argNum++
					i++
					continue formatLoop
				}
				// Format is more complex than simple flags and a verb or is malformed.
				break simpleFormat
			}
		}

		// Do we have an explicit argument index?
		i, afterIndex = p.updateArgNumber(format, i)

		// Do we have width?
		if i < end && format[i] == '*' {
			i++
			p.fmt.wid, p.fmt.widPresent = p.intFromArg()

			if !p.fmt.widPresent {
				p.WriteString(badWidthString)
			}

			// We have a negative width, so take its value and ensure
			// that the minus flag is set
			if p.fmt.wid < 0 {
				p.fmt.wid = -p.fmt.wid
				p.fmt.minus = true
				p.fmt.zero = false // Do not pad with zeros to the right.
			}
			afterIndex = false
		} else {
			p.fmt.wid, p.fmt.widPresent, i = parsenum(format, i, end)
			if afterIndex && p.fmt.widPresent { // "%[3]2d"
				p.goodArgNum = false
			}
		}

		// Do we have precision?
		if i+1 < end && format[i] == '.' {
			i++
			if afterIndex { // "%[3].2d"
				p.goodArgNum = false
			}
			i, afterIndex = p.updateArgNumber(format, i)
			if i < end && format[i] == '*' {
				i++
				p.fmt.prec, p.fmt.precPresent = p.intFromArg()
				// Negative precision arguments don't make sense
				if p.fmt.prec < 0 {
					p.fmt.prec = 0
					p.fmt.precPresent = false
				}
				if !p.fmt.precPresent {
					p.WriteString(badPrecString)
				}
				afterIndex = false
			} else {
				p.fmt.prec, p.fmt.precPresent, i = parsenum(format, i, end)
				if !p.fmt.precPresent {
					p.fmt.prec = 0
					p.fmt.precPresent = true
				}
			}
		}

		if !afterIndex {
			i, afterIndex = p.updateArgNumber(format, i)
		}

		if i >= end {
			p.WriteString(noVerbString)
			break
		}

		verb, w := utf8.DecodeRuneInString(format[i:])
		i += w

		switch {
		case verb == '%': // Percent does not absorb operands and ignores f.wid and f.prec.
			p.WriteByte('%')
		case !p.goodArgNum:
			p.badArgNum(verb)
		case p.argNum >= len(p.args): // No argument left over to print for the current verb.
			p.missingArg(verb)
		case verb == 'v':
			// Go syntax
			p.fmt.sharpV = p.fmt.sharp
			p.fmt.sharp = false
			// Struct-field syntax
			p.fmt.plusV = p.fmt.plus
			p.fmt.plus = false
			fallthrough
		default:
			p.printArg(p.args[p.argNum], verb)
			p.argNum++
		}
	}

	// Check for extra arguments, but only if there was at least one ordered
	// argument. Note that this behavior is necessarily different from fmt:
	// different variants of messages may opt to drop some or all of the
	// arguments.
	if !p.reordered && p.argNum < len(p.args) && p.argNum != 0 {
		p.fmt.clearflags()
		p.WriteString(extraString)
		for i, arg := range p.args[p.argNum:] {
			if i > 0 {
				p.WriteString(commaSpaceString)
			}
			if arg == nil {
				p.WriteString(nilAngleString)
			} else {
				p.WriteString(reflect.TypeOf(arg).String())
				p.WriteByte('=')
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
