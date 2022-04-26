// Package staticcheck contains a linter for Go source code.
package staticcheck

import (
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	htmltemplate "html/template"
	"net/http"
	"reflect"
	"regexp"
	"regexp/syntax"
	"sort"
	"strconv"
	"strings"
	texttemplate "text/template"
	"unicode"

	"honnef.co/go/tools/analysis/code"
	"honnef.co/go/tools/analysis/edit"
	"honnef.co/go/tools/analysis/facts"
	"honnef.co/go/tools/analysis/facts/nilness"
	"honnef.co/go/tools/analysis/facts/typedness"
	"honnef.co/go/tools/analysis/lint"
	"honnef.co/go/tools/analysis/report"
	"honnef.co/go/tools/go/ast/astutil"
	"honnef.co/go/tools/go/ir"
	"honnef.co/go/tools/go/ir/irutil"
	"honnef.co/go/tools/go/types/typeutil"
	"honnef.co/go/tools/internal/passes/buildir"
	"honnef.co/go/tools/internal/sharedcheck"
	"honnef.co/go/tools/knowledge"
	"honnef.co/go/tools/pattern"
	"honnef.co/go/tools/printf"
	"honnef.co/go/tools/staticcheck/fakejson"
	"honnef.co/go/tools/staticcheck/fakereflect"
	"honnef.co/go/tools/staticcheck/fakexml"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
)

func checkSortSlice(call *Call) {
	c := call.Instr.Common().StaticCallee()
	arg := call.Args[0]

	T := arg.Value.Value.Type().Underlying()
	switch T.(type) {
	case *types.Interface:
		// we don't know.
		// TODO(dh): if the value is a phi node we can look at its edges
		if k, ok := arg.Value.Value.(*ir.Const); ok && k.Value == nil {
			// literal nil, e.g. sort.Sort(nil, ...)
			arg.Invalid(fmt.Sprintf("cannot call %s on nil literal", c))
		}
	case *types.Slice:
		// this is fine
	default:
		// this is not fine
		arg.Invalid(fmt.Sprintf("%s must only be called on slices, was called on %s", c, T))
	}
}

func validRegexp(call *Call) {
	arg := call.Args[0]
	err := ValidateRegexp(arg.Value)
	if err != nil {
		arg.Invalid(err.Error())
	}
}

type runeSlice []rune

func (rs runeSlice) Len() int               { return len(rs) }
func (rs runeSlice) Less(i int, j int) bool { return rs[i] < rs[j] }
func (rs runeSlice) Swap(i int, j int)      { rs[i], rs[j] = rs[j], rs[i] }

func utf8Cutset(call *Call) {
	arg := call.Args[1]
	if InvalidUTF8(arg.Value) {
		arg.Invalid(MsgInvalidUTF8)
	}
}

func uniqueCutset(call *Call) {
	arg := call.Args[1]
	if !UniqueStringCutset(arg.Value) {
		arg.Invalid(MsgNonUniqueCutset)
	}
}

func unmarshalPointer(name string, arg int) CallCheck {
	return func(call *Call) {
		if !Pointer(call.Args[arg].Value) {
			call.Args[arg].Invalid(fmt.Sprintf("%s expects to unmarshal into a pointer, but the provided value is not a pointer", name))
		}
	}
}

func pointlessIntMath(call *Call) {
	if ConvertedFromInt(call.Args[0].Value) {
		call.Invalid(fmt.Sprintf("calling %s on a converted integer is pointless", irutil.CallName(call.Instr.Common())))
	}
}

func checkValidHostPort(arg int) CallCheck {
	return func(call *Call) {
		if !ValidHostPort(call.Args[arg].Value) {
			call.Args[arg].Invalid(MsgInvalidHostPort)
		}
	}
}

var (
	checkRegexpRules = map[string]CallCheck{
		"regexp.MustCompile": validRegexp,
		"regexp.Compile":     validRegexp,
		"regexp.Match":       validRegexp,
		"regexp.MatchReader": validRegexp,
		"regexp.MatchString": validRegexp,
	}

	checkTimeParseRules = map[string]CallCheck{
		"time.Parse": func(call *Call) {
			arg := call.Args[knowledge.Arg("time.Parse.layout")]
			err := ValidateTimeLayout(arg.Value)
			if err != nil {
				arg.Invalid(err.Error())
			}
		},
	}

	checkEncodingBinaryRules = map[string]CallCheck{
		"encoding/binary.Write": func(call *Call) {
			arg := call.Args[knowledge.Arg("encoding/binary.Write.data")]
			if !CanBinaryMarshal(call.Pass, arg.Value) {
				arg.Invalid(fmt.Sprintf("value of type %s cannot be used with binary.Write", arg.Value.Value.Type()))
			}
		},
	}

	checkURLsRules = map[string]CallCheck{
		"net/url.Parse": func(call *Call) {
			arg := call.Args[knowledge.Arg("net/url.Parse.rawurl")]
			err := ValidateURL(arg.Value)
			if err != nil {
				arg.Invalid(err.Error())
			}
		},
	}

	checkSyncPoolValueRules = map[string]CallCheck{
		"(*sync.Pool).Put": func(call *Call) {
			arg := call.Args[knowledge.Arg("(*sync.Pool).Put.x")]
			typ := arg.Value.Value.Type()
			_, isSlice := typ.Underlying().(*types.Slice)
			if !typeutil.IsPointerLike(typ) || isSlice {
				arg.Invalid("argument should be pointer-like to avoid allocations")
			}
		},
	}

	checkRegexpFindAllRules = map[string]CallCheck{
		"(*regexp.Regexp).FindAll":                    RepeatZeroTimes("a FindAll method", 1),
		"(*regexp.Regexp).FindAllIndex":               RepeatZeroTimes("a FindAll method", 1),
		"(*regexp.Regexp).FindAllString":              RepeatZeroTimes("a FindAll method", 1),
		"(*regexp.Regexp).FindAllStringIndex":         RepeatZeroTimes("a FindAll method", 1),
		"(*regexp.Regexp).FindAllStringSubmatch":      RepeatZeroTimes("a FindAll method", 1),
		"(*regexp.Regexp).FindAllStringSubmatchIndex": RepeatZeroTimes("a FindAll method", 1),
		"(*regexp.Regexp).FindAllSubmatch":            RepeatZeroTimes("a FindAll method", 1),
		"(*regexp.Regexp).FindAllSubmatchIndex":       RepeatZeroTimes("a FindAll method", 1),
	}

	checkUTF8CutsetRules = map[string]CallCheck{
		"strings.IndexAny":     utf8Cutset,
		"strings.LastIndexAny": utf8Cutset,
		"strings.ContainsAny":  utf8Cutset,
		"strings.Trim":         utf8Cutset,
		"strings.TrimLeft":     utf8Cutset,
		"strings.TrimRight":    utf8Cutset,
	}

	checkUniqueCutsetRules = map[string]CallCheck{
		"strings.Trim":      uniqueCutset,
		"strings.TrimLeft":  uniqueCutset,
		"strings.TrimRight": uniqueCutset,
	}

	checkUnmarshalPointerRules = map[string]CallCheck{
		"encoding/xml.Unmarshal":                unmarshalPointer("xml.Unmarshal", 1),
		"(*encoding/xml.Decoder).Decode":        unmarshalPointer("Decode", 0),
		"(*encoding/xml.Decoder).DecodeElement": unmarshalPointer("DecodeElement", 0),
		"encoding/json.Unmarshal":               unmarshalPointer("json.Unmarshal", 1),
		"(*encoding/json.Decoder).Decode":       unmarshalPointer("Decode", 0),
	}

	checkUnbufferedSignalChanRules = map[string]CallCheck{
		"os/signal.Notify": func(call *Call) {
			arg := call.Args[knowledge.Arg("os/signal.Notify.c")]
			if UnbufferedChannel(arg.Value) {
				arg.Invalid("the channel used with signal.Notify should be buffered")
			}
		},
	}

	checkMathIntRules = map[string]CallCheck{
		"math.Ceil":  pointlessIntMath,
		"math.Floor": pointlessIntMath,
		"math.IsNaN": pointlessIntMath,
		"math.Trunc": pointlessIntMath,
		"math.IsInf": pointlessIntMath,
	}

	checkStringsReplaceZeroRules = map[string]CallCheck{
		"strings.Replace": RepeatZeroTimes("strings.Replace", 3),
		"bytes.Replace":   RepeatZeroTimes("bytes.Replace", 3),
	}

	checkListenAddressRules = map[string]CallCheck{
		"net/http.ListenAndServe":    checkValidHostPort(0),
		"net/http.ListenAndServeTLS": checkValidHostPort(0),
	}

	checkBytesEqualIPRules = map[string]CallCheck{
		"bytes.Equal": func(call *Call) {
			if ConvertedFrom(call.Args[knowledge.Arg("bytes.Equal.a")].Value, "net.IP") &&
				ConvertedFrom(call.Args[knowledge.Arg("bytes.Equal.b")].Value, "net.IP") {
				call.Invalid("use net.IP.Equal to compare net.IPs, not bytes.Equal")
			}
		},
	}

	checkRegexpMatchLoopRules = map[string]CallCheck{
		"regexp.Match":       loopedRegexp("regexp.Match"),
		"regexp.MatchReader": loopedRegexp("regexp.MatchReader"),
		"regexp.MatchString": loopedRegexp("regexp.MatchString"),
	}

	checkNoopMarshal = map[string]CallCheck{
		// TODO(dh): should we really flag XML? Even an empty struct
		// produces a non-zero amount of data, namely its type name.
		// Let's see if we encounter any false positives.
		//
		// Also, should we flag gob?
		"encoding/json.Marshal":           checkNoopMarshalImpl(knowledge.Arg("json.Marshal.v"), "MarshalJSON", "MarshalText"),
		"encoding/xml.Marshal":            checkNoopMarshalImpl(knowledge.Arg("xml.Marshal.v"), "MarshalXML", "MarshalText"),
		"(*encoding/json.Encoder).Encode": checkNoopMarshalImpl(knowledge.Arg("(*encoding/json.Encoder).Encode.v"), "MarshalJSON", "MarshalText"),
		"(*encoding/xml.Encoder).Encode":  checkNoopMarshalImpl(knowledge.Arg("(*encoding/xml.Encoder).Encode.v"), "MarshalXML", "MarshalText"),

		"encoding/json.Unmarshal":         checkNoopMarshalImpl(knowledge.Arg("json.Unmarshal.v"), "UnmarshalJSON", "UnmarshalText"),
		"encoding/xml.Unmarshal":          checkNoopMarshalImpl(knowledge.Arg("xml.Unmarshal.v"), "UnmarshalXML", "UnmarshalText"),
		"(*encoding/json.Decoder).Decode": checkNoopMarshalImpl(knowledge.Arg("(*encoding/json.Decoder).Decode.v"), "UnmarshalJSON", "UnmarshalText"),
		"(*encoding/xml.Decoder).Decode":  checkNoopMarshalImpl(knowledge.Arg("(*encoding/xml.Decoder).Decode.v"), "UnmarshalXML", "UnmarshalText"),
	}

	checkUnsupportedMarshal = map[string]CallCheck{
		"encoding/json.Marshal":           checkUnsupportedMarshalJSON,
		"encoding/xml.Marshal":            checkUnsupportedMarshalXML,
		"(*encoding/json.Encoder).Encode": checkUnsupportedMarshalJSON,
		"(*encoding/xml.Encoder).Encode":  checkUnsupportedMarshalXML,
	}

	checkAtomicAlignment = map[string]CallCheck{
		"sync/atomic.AddInt64":             checkAtomicAlignmentImpl,
		"sync/atomic.AddUint64":            checkAtomicAlignmentImpl,
		"sync/atomic.CompareAndSwapInt64":  checkAtomicAlignmentImpl,
		"sync/atomic.CompareAndSwapUint64": checkAtomicAlignmentImpl,
		"sync/atomic.LoadInt64":            checkAtomicAlignmentImpl,
		"sync/atomic.LoadUint64":           checkAtomicAlignmentImpl,
		"sync/atomic.StoreInt64":           checkAtomicAlignmentImpl,
		"sync/atomic.StoreUint64":          checkAtomicAlignmentImpl,
		"sync/atomic.SwapInt64":            checkAtomicAlignmentImpl,
		"sync/atomic.SwapUint64":           checkAtomicAlignmentImpl,
	}

	// TODO(dh): detect printf wrappers
	checkPrintfRules = map[string]CallCheck{
		"fmt.Errorf":                  func(call *Call) { checkPrintfCall(call, 0, 1) },
		"fmt.Printf":                  func(call *Call) { checkPrintfCall(call, 0, 1) },
		"fmt.Sprintf":                 func(call *Call) { checkPrintfCall(call, 0, 1) },
		"fmt.Fprintf":                 func(call *Call) { checkPrintfCall(call, 1, 2) },
		"golang.org/x/xerrors.Errorf": func(call *Call) { checkPrintfCall(call, 0, 1) },
	}

	checkSortSliceRules = map[string]CallCheck{
		"sort.Slice":         checkSortSlice,
		"sort.SliceIsSorted": checkSortSlice,
		"sort.SliceStable":   checkSortSlice,
	}

	checkWithValueKeyRules = map[string]CallCheck{
		"context.WithValue": checkWithValueKey,
	}

	checkStrconvRules = map[string]CallCheck{
		"strconv.ParseComplex": func(call *Call) {
			validateComplexBitSize(call.Args[knowledge.Arg("strconv.ParseComplex.bitSize")])
		},
		"strconv.ParseFloat": func(call *Call) {
			validateFloatBitSize(call.Args[knowledge.Arg("strconv.ParseFloat.bitSize")])
		},
		"strconv.ParseInt": func(call *Call) {
			validateContinuousBitSize(call.Args[knowledge.Arg("strconv.ParseInt.bitSize")], 0, 64)
			validateIntBaseAllowZero(call.Args[knowledge.Arg("strconv.ParseInt.base")])
		},
		"strconv.ParseUint": func(call *Call) {
			validateContinuousBitSize(call.Args[knowledge.Arg("strconv.ParseUint.bitSize")], 0, 64)
			validateIntBaseAllowZero(call.Args[knowledge.Arg("strconv.ParseUint.base")])
		},

		"strconv.FormatComplex": func(call *Call) {
			validateComplexFormat(call.Args[knowledge.Arg("strconv.FormatComplex.fmt")])
			validateComplexBitSize(call.Args[knowledge.Arg("strconv.FormatComplex.bitSize")])
		},
		"strconv.FormatFloat": func(call *Call) {
			validateFloatFormat(call.Args[knowledge.Arg("strconv.FormatFloat.fmt")])
			validateFloatBitSize(call.Args[knowledge.Arg("strconv.FormatFloat.bitSize")])
		},
		"strconv.FormatInt": func(call *Call) {
			validateIntBase(call.Args[knowledge.Arg("strconv.FormatInt.base")])
		},
		"strconv.FormatUint": func(call *Call) {
			validateIntBase(call.Args[knowledge.Arg("strconv.FormatUint.base")])
		},

		"strconv.AppendFloat": func(call *Call) {
			validateFloatFormat(call.Args[knowledge.Arg("strconv.AppendFloat.fmt")])
			validateFloatBitSize(call.Args[knowledge.Arg("strconv.AppendFloat.bitSize")])
		},
		"strconv.AppendInt": func(call *Call) {
			validateIntBase(call.Args[knowledge.Arg("strconv.AppendInt.base")])
		},
		"strconv.AppendUint": func(call *Call) {
			validateIntBase(call.Args[knowledge.Arg("strconv.AppendUint.base")])
		},
	}
)

func validateIntBase(arg *Argument) {
	if c := extractConstExpectKind(arg.Value.Value, constant.Int); c != nil {
		val, _ := constant.Int64Val(c.Value)
		if val < 2 {
			arg.Invalid("'base' must not be smaller than 2")
		}
		if val > 36 {
			arg.Invalid("'base' must not be larger than 36")
		}
	}
}

func validateIntBaseAllowZero(arg *Argument) {
	if c := extractConstExpectKind(arg.Value.Value, constant.Int); c != nil {
		val, _ := constant.Int64Val(c.Value)
		if val < 2 && val != 0 {
			arg.Invalid("'base' must not be smaller than 2, unless it is 0")
		}
		if val > 36 {
			arg.Invalid("'base' must not be larger than 36")
		}
	}
}

func validateComplexFormat(arg *Argument) {
	validateFloatFormat(arg)
}

func validateFloatFormat(arg *Argument) {
	if c := extractConstExpectKind(arg.Value.Value, constant.Int); c != nil {
		val, _ := constant.Int64Val(c.Value)
		switch val {
		case 'b', 'e', 'E', 'f', 'g', 'G', 'x', 'X':
		default:
			arg.Invalid(fmt.Sprintf("'fmt' argument is invalid: unknown format %q", val))
		}
	}
}

func validateComplexBitSize(arg *Argument) { validateDiscreetBitSize(arg, 64, 128) }
func validateFloatBitSize(arg *Argument)   { validateDiscreetBitSize(arg, 32, 64) }

func validateDiscreetBitSize(arg *Argument, size1 int, size2 int) {
	if c := extractConstExpectKind(arg.Value.Value, constant.Int); c != nil {
		val, _ := constant.Int64Val(c.Value)
		if val != int64(size1) && val != int64(size2) {
			arg.Invalid(fmt.Sprintf("'bitSize' argument is invalid, must be either %d or %d", size1, size2))
		}
	}
}

func validateContinuousBitSize(arg *Argument, min int, max int) {
	if c := extractConstExpectKind(arg.Value.Value, constant.Int); c != nil {
		val, _ := constant.Int64Val(c.Value)
		if val < int64(min) || val > int64(max) {
			arg.Invalid(fmt.Sprintf("'bitSize' argument is invalid, must be within %d and %d", min, max))
		}
	}
}

func checkPrintfCall(call *Call, fIdx, vIdx int) {
	f := call.Args[fIdx]
	var args []ir.Value
	switch v := call.Args[vIdx].Value.Value.(type) {
	case *ir.Slice:
		var ok bool
		args, ok = irutil.Vararg(v)
		if !ok {
			// We don't know what the actual arguments to the function are
			return
		}
	case *ir.Const:
		// nil, i.e. no arguments
	default:
		// We don't know what the actual arguments to the function are
		return
	}
	checkPrintfCallImpl(f, f.Value.Value, args)
}

type verbFlag int

const (
	isInt verbFlag = 1 << iota
	isBool
	isFP
	isString
	isPointer
	// Verbs that accept "pseudo pointers" will sometimes dereference
	// non-nil pointers. For example, %x on a non-nil *struct will print the
	// individual fields, but on a nil pointer it will print the address.
	isPseudoPointer
	isSlice
	isAny
	noRecurse
)

var verbs = [...]verbFlag{
	'b': isPseudoPointer | isInt | isFP,
	'c': isInt,
	'd': isPseudoPointer | isInt,
	'e': isFP,
	'E': isFP,
	'f': isFP,
	'F': isFP,
	'g': isFP,
	'G': isFP,
	'o': isPseudoPointer | isInt,
	'O': isPseudoPointer | isInt,
	'p': isSlice | isPointer | noRecurse,
	'q': isInt | isString,
	's': isString,
	't': isBool,
	'T': isAny,
	'U': isInt,
	'v': isAny,
	'X': isPseudoPointer | isInt | isFP | isString,
	'x': isPseudoPointer | isInt | isFP | isString,
}

func checkPrintfCallImpl(carg *Argument, f ir.Value, args []ir.Value) {
	var msCache *typeutil.MethodSetCache
	if f.Parent() != nil {
		msCache = &f.Parent().Prog.MethodSets
	}

	elem := func(T types.Type, verb rune) ([]types.Type, bool) {
		if verbs[verb]&noRecurse != 0 {
			return []types.Type{T}, false
		}
		switch T := T.(type) {
		case *types.Slice:
			if verbs[verb]&isSlice != 0 {
				return []types.Type{T}, false
			}
			if verbs[verb]&isString != 0 && typeutil.IsType(T.Elem().Underlying(), "byte") {
				return []types.Type{T}, false
			}
			return []types.Type{T.Elem()}, true
		case *types.Map:
			key := T.Key()
			val := T.Elem()
			return []types.Type{key, val}, true
		case *types.Struct:
			out := make([]types.Type, 0, T.NumFields())
			for i := 0; i < T.NumFields(); i++ {
				out = append(out, T.Field(i).Type())
			}
			return out, true
		case *types.Array:
			return []types.Type{T.Elem()}, true
		default:
			return []types.Type{T}, false
		}
	}
	isInfo := func(T types.Type, info types.BasicInfo) bool {
		basic, ok := T.Underlying().(*types.Basic)
		return ok && basic.Info()&info != 0
	}

	isStringer := func(T types.Type, ms *types.MethodSet) bool {
		sel := ms.Lookup(nil, "String")
		if sel == nil {
			return false
		}
		fn, ok := sel.Obj().(*types.Func)
		if !ok {
			// should be unreachable
			return false
		}
		sig := fn.Type().(*types.Signature)
		if sig.Params().Len() != 0 {
			return false
		}
		if sig.Results().Len() != 1 {
			return false
		}
		if !typeutil.IsType(sig.Results().At(0).Type(), "string") {
			return false
		}
		return true
	}
	isError := func(T types.Type, ms *types.MethodSet) bool {
		sel := ms.Lookup(nil, "Error")
		if sel == nil {
			return false
		}
		fn, ok := sel.Obj().(*types.Func)
		if !ok {
			// should be unreachable
			return false
		}
		sig := fn.Type().(*types.Signature)
		if sig.Params().Len() != 0 {
			return false
		}
		if sig.Results().Len() != 1 {
			return false
		}
		if !typeutil.IsType(sig.Results().At(0).Type(), "string") {
			return false
		}
		return true
	}

	isFormatter := func(T types.Type, ms *types.MethodSet) bool {
		sel := ms.Lookup(nil, "Format")
		if sel == nil {
			return false
		}
		fn, ok := sel.Obj().(*types.Func)
		if !ok {
			// should be unreachable
			return false
		}
		sig := fn.Type().(*types.Signature)
		if sig.Params().Len() != 2 {
			return false
		}
		// TODO(dh): check the types of the arguments for more
		// precision
		if sig.Results().Len() != 0 {
			return false
		}
		return true
	}

	seen := map[types.Type]bool{}
	var checkType func(verb rune, T types.Type, top bool) bool
	checkType = func(verb rune, T types.Type, top bool) bool {
		if top {
			for k := range seen {
				delete(seen, k)
			}
		}
		if seen[T] {
			return true
		}
		seen[T] = true
		if int(verb) >= len(verbs) {
			// Unknown verb
			return true
		}

		flags := verbs[verb]
		if flags == 0 {
			// Unknown verb
			return true
		}

		ms := msCache.MethodSet(T)
		if isFormatter(T, ms) {
			// the value is responsible for formatting itself
			return true
		}

		if flags&isString != 0 && (isStringer(T, ms) || isError(T, ms)) {
			// Check for stringer early because we're about to dereference
			return true
		}

		T = T.Underlying()
		if flags&(isPointer|isPseudoPointer) == 0 && top {
			T = typeutil.Dereference(T)
		}
		if flags&isPseudoPointer != 0 && top {
			t := typeutil.Dereference(T)
			if _, ok := t.Underlying().(*types.Struct); ok {
				T = t
			}
		}

		if _, ok := T.(*types.Interface); ok {
			// We don't know what's in the interface
			return true
		}

		var info types.BasicInfo
		if flags&isInt != 0 {
			info |= types.IsInteger
		}
		if flags&isBool != 0 {
			info |= types.IsBoolean
		}
		if flags&isFP != 0 {
			info |= types.IsFloat | types.IsComplex
		}
		if flags&isString != 0 {
			info |= types.IsString
		}

		if info != 0 && isInfo(T, info) {
			return true
		}

		if flags&isString != 0 {
			isStringyElem := func(typ types.Type) bool {
				if typ, ok := typ.Underlying().(*types.Basic); ok {
					return typ.Kind() == types.Byte
				}
				return false
			}
			switch T := T.(type) {
			case *types.Slice:
				if isStringyElem(T.Elem()) {
					return true
				}
			case *types.Array:
				if isStringyElem(T.Elem()) {
					return true
				}
			}
			if isStringer(T, ms) || isError(T, ms) {
				return true
			}
		}

		if flags&isPointer != 0 && typeutil.IsPointerLike(T) {
			return true
		}
		if flags&isPseudoPointer != 0 {
			switch U := T.Underlying().(type) {
			case *types.Pointer:
				if !top {
					return true
				}

				if _, ok := U.Elem().Underlying().(*types.Struct); !ok {
					// TODO(dh): can this condition ever be false? For
					// *T, if T is a struct, we'll already have
					// dereferenced it, meaning the *types.Pointer
					// branch couldn't have been taken. For T that
					// aren't structs, this condition will always
					// evaluate to true.
					return true
				}
			case *types.Chan, *types.Signature:
				// Channels and functions are always treated as
				// pointers and never recursed into.
				return true
			case *types.Basic:
				if U.Kind() == types.UnsafePointer {
					return true
				}
			case *types.Interface:
				// we will already have bailed if the type is an
				// interface.
				panic("unreachable")
			default:
				// other pointer-like types, such as maps or slices,
				// will be printed element-wise.
			}
		}

		if flags&isSlice != 0 {
			if _, ok := T.(*types.Slice); ok {
				return true
			}
		}

		if flags&isAny != 0 {
			return true
		}

		elems, ok := elem(T.Underlying(), verb)
		if !ok {
			return false
		}
		for _, elem := range elems {
			if !checkType(verb, elem, false) {
				return false
			}
		}

		return true
	}

	k, ok := irutil.Flatten(f).(*ir.Const)
	if !ok {
		return
	}
	actions, err := printf.Parse(constant.StringVal(k.Value))
	if err != nil {
		carg.Invalid("couldn't parse format string")
		return
	}

	ptr := 1
	hasExplicit := false

	checkStar := func(verb printf.Verb, star printf.Argument) bool {
		if star, ok := star.(printf.Star); ok {
			idx := 0
			if star.Index == -1 {
				idx = ptr
				ptr++
			} else {
				hasExplicit = true
				idx = star.Index
				ptr = star.Index + 1
			}
			if idx == 0 {
				carg.Invalid(fmt.Sprintf("Printf format %s reads invalid arg 0; indices are 1-based", verb.Raw))
				return false
			}
			if idx > len(args) {
				carg.Invalid(
					fmt.Sprintf("Printf format %s reads arg #%d, but call has only %d args",
						verb.Raw, idx, len(args)))
				return false
			}
			if arg, ok := args[idx-1].(*ir.MakeInterface); ok {
				if !isInfo(arg.X.Type(), types.IsInteger) {
					carg.Invalid(fmt.Sprintf("Printf format %s reads non-int arg #%d as argument of *", verb.Raw, idx))
				}
			}
		}
		return true
	}

	// We only report one problem per format string. Making a
	// mistake with an index tends to invalidate all future
	// implicit indices.
	for _, action := range actions {
		verb, ok := action.(printf.Verb)
		if !ok {
			continue
		}

		if !checkStar(verb, verb.Width) || !checkStar(verb, verb.Precision) {
			return
		}

		off := ptr
		if verb.Value != -1 {
			hasExplicit = true
			off = verb.Value
		}
		if off > len(args) {
			carg.Invalid(
				fmt.Sprintf("Printf format %s reads arg #%d, but call has only %d args",
					verb.Raw, off, len(args)))
			return
		} else if verb.Value == 0 && verb.Letter != '%' {
			carg.Invalid(fmt.Sprintf("Printf format %s reads invalid arg 0; indices are 1-based", verb.Raw))
			return
		} else if off != 0 {
			arg, ok := args[off-1].(*ir.MakeInterface)
			if ok {
				if !checkType(verb.Letter, arg.X.Type(), true) {
					carg.Invalid(fmt.Sprintf("Printf format %s has arg #%d of wrong type %s",
						verb.Raw, ptr, args[ptr-1].(*ir.MakeInterface).X.Type()))
					return
				}
			}
		}

		switch verb.Value {
		case -1:
			// Consume next argument
			ptr++
		case 0:
			// Don't consume any arguments
		default:
			ptr = verb.Value + 1
		}
	}

	if !hasExplicit && ptr <= len(args) {
		carg.Invalid(fmt.Sprintf("Printf call needs %d args but has %d args", ptr-1, len(args)))
	}
}

func checkAtomicAlignmentImpl(call *Call) {
	sizes := call.Pass.TypesSizes
	if sizes.Sizeof(types.Typ[types.Uintptr]) != 4 {
		// Not running on a 32-bit platform
		return
	}
	v, ok := irutil.Flatten(call.Args[0].Value.Value).(*ir.FieldAddr)
	if !ok {
		// TODO(dh): also check indexing into arrays and slices
		return
	}
	T := v.X.Type().Underlying().(*types.Pointer).Elem().Underlying().(*types.Struct)
	fields := make([]*types.Var, 0, T.NumFields())
	for i := 0; i < T.NumFields() && i <= v.Field; i++ {
		fields = append(fields, T.Field(i))
	}

	off := sizes.Offsetsof(fields)[v.Field]
	if off%8 != 0 {
		msg := fmt.Sprintf("address of non 64-bit aligned field %s passed to %s",
			T.Field(v.Field).Name(),
			irutil.CallName(call.Instr.Common()))
		call.Invalid(msg)
	}
}

func checkNoopMarshalImpl(argN int, meths ...string) CallCheck {
	return func(call *Call) {
		if code.IsGenerated(call.Pass, call.Instr.Pos()) {
			return
		}
		arg := call.Args[argN]
		T := arg.Value.Value.Type()
		Ts, ok := typeutil.Dereference(T).Underlying().(*types.Struct)
		if !ok {
			return
		}
		if Ts.NumFields() == 0 {
			return
		}
		fields := typeutil.FlattenFields(Ts)
		for _, field := range fields {
			if field.Var.Exported() {
				return
			}
		}
		// OPT(dh): we could use a method set cache here
		ms := call.Instr.Parent().Prog.MethodSets.MethodSet(T)
		// TODO(dh): we're not checking the signature, which can cause false negatives.
		// This isn't a huge problem, however, since vet complains about incorrect signatures.
		for _, meth := range meths {
			if ms.Lookup(nil, meth) != nil {
				return
			}
		}
		arg.Invalid(fmt.Sprintf("struct type '%s' doesn't have any exported fields, nor custom marshaling", typeutil.Dereference(T)))
	}
}

func checkUnsupportedMarshalJSON(call *Call) {
	arg := call.Args[0]
	T := arg.Value.Value.Type()
	if err := fakejson.Marshal(T); err != nil {
		typ := types.TypeString(err.Type, types.RelativeTo(arg.Value.Value.Parent().Pkg.Pkg))
		if err.Path == "x" {
			arg.Invalid(fmt.Sprintf("trying to marshal unsupported type %s", typ))
		} else {
			arg.Invalid(fmt.Sprintf("trying to marshal unsupported type %s, via %s", typ, err.Path))
		}
	}
}

func checkUnsupportedMarshalXML(call *Call) {
	arg := call.Args[0]
	T := arg.Value.Value.Type()
	if err := fakexml.Marshal(T); err != nil {
		switch err := err.(type) {
		case *fakexml.UnsupportedTypeError:
			typ := types.TypeString(err.Type, types.RelativeTo(arg.Value.Value.Parent().Pkg.Pkg))
			if err.Path == "x" {
				arg.Invalid(fmt.Sprintf("trying to marshal unsupported type %s", typ))
			} else {
				arg.Invalid(fmt.Sprintf("trying to marshal unsupported type %s, via %s", typ, err.Path))
			}
		case *fakexml.TagPathError:
			// Vet does a better job at reporting this error, because it can flag the actual struct tags, not just the call to Marshal
		default:
			// These errors get reported by SA5008 instead, which can flag the actual fields, independently of calls to xml.Marshal
		}
	}
}

func isInLoop(b *ir.BasicBlock) bool {
	sets := irutil.FindLoops(b.Parent())
	for _, set := range sets {
		if set.Has(b) {
			return true
		}
	}
	return false
}

func CheckUntrappableSignal(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		call := node.(*ast.CallExpr)
		if !code.IsCallToAny(pass, call,
			"os/signal.Ignore", "os/signal.Notify", "os/signal.Reset") {
			return
		}

		hasSigterm := false
		for _, arg := range call.Args {
			if conv, ok := arg.(*ast.CallExpr); ok && isName(pass, conv.Fun, "os.Signal") {
				arg = conv.Args[0]
			}

			if isName(pass, arg, "syscall.SIGTERM") {
				hasSigterm = true
				break
			}

		}
		for i, arg := range call.Args {
			if conv, ok := arg.(*ast.CallExpr); ok && isName(pass, conv.Fun, "os.Signal") {
				arg = conv.Args[0]
			}

			if isName(pass, arg, "os.Kill") || isName(pass, arg, "syscall.SIGKILL") {
				var fixes []analysis.SuggestedFix
				if !hasSigterm {
					nargs := make([]ast.Expr, len(call.Args))
					for j, a := range call.Args {
						if i == j {
							nargs[j] = edit.Selector("syscall", "SIGTERM")
						} else {
							nargs[j] = a
						}
					}
					ncall := *call
					ncall.Args = nargs
					fixes = append(fixes, edit.Fix(fmt.Sprintf("use syscall.SIGTERM instead of %s", report.Render(pass, arg)), edit.ReplaceWithNode(pass.Fset, call, &ncall)))
				}
				nargs := make([]ast.Expr, 0, len(call.Args))
				for j, a := range call.Args {
					if i == j {
						continue
					}
					nargs = append(nargs, a)
				}
				ncall := *call
				ncall.Args = nargs
				fixes = append(fixes, edit.Fix(fmt.Sprintf("remove %s from list of arguments", report.Render(pass, arg)), edit.ReplaceWithNode(pass.Fset, call, &ncall)))
				report.Report(pass, arg, fmt.Sprintf("%s cannot be trapped (did you mean syscall.SIGTERM?)", report.Render(pass, arg)), report.Fixes(fixes...))
			}
			if isName(pass, arg, "syscall.SIGSTOP") {
				nargs := make([]ast.Expr, 0, len(call.Args)-1)
				for j, a := range call.Args {
					if i == j {
						continue
					}
					nargs = append(nargs, a)
				}
				ncall := *call
				ncall.Args = nargs
				report.Report(pass, arg, "syscall.SIGSTOP cannot be trapped", report.Fixes(edit.Fix("remove syscall.SIGSTOP from list of arguments", edit.ReplaceWithNode(pass.Fset, call, &ncall))))
			}
		}
	}
	code.Preorder(pass, fn, (*ast.CallExpr)(nil))
	return nil, nil
}

func CheckTemplate(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		call := node.(*ast.CallExpr)
		// OPT(dh): use integer for kind
		var kind string
		switch code.CallName(pass, call) {
		case "(*text/template.Template).Parse":
			kind = "text"
		case "(*html/template.Template).Parse":
			kind = "html"
		default:
			return
		}
		sel := call.Fun.(*ast.SelectorExpr)
		if !code.IsCallToAny(pass, sel.X, "text/template.New", "html/template.New") {
			// TODO(dh): this is a cheap workaround for templates with
			// different delims. A better solution with less false
			// negatives would use data flow analysis to see where the
			// template comes from and where it has been
			return
		}
		s, ok := code.ExprToString(pass, call.Args[knowledge.Arg("(*text/template.Template).Parse.text")])
		if !ok {
			return
		}
		var err error
		switch kind {
		case "text":
			_, err = texttemplate.New("").Parse(s)
		case "html":
			_, err = htmltemplate.New("").Parse(s)
		}
		if err != nil {
			// TODO(dominikh): whitelist other parse errors, if any
			if strings.Contains(err.Error(), "unexpected") {
				report.Report(pass, call.Args[knowledge.Arg("(*text/template.Template).Parse.text")], err.Error())
			}
		}
	}
	code.Preorder(pass, fn, (*ast.CallExpr)(nil))
	return nil, nil
}

var (
	checkTimeSleepConstantPatternRns = pattern.MustParse(`(BinaryExpr duration "*" (SelectorExpr (Ident "time") (Ident "Nanosecond")))`)
	checkTimeSleepConstantPatternRs  = pattern.MustParse(`(BinaryExpr duration "*" (SelectorExpr (Ident "time") (Ident "Second")))`)
)

func CheckTimeSleepConstant(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		call := node.(*ast.CallExpr)
		if !code.IsCallTo(pass, call, "time.Sleep") {
			return
		}
		lit, ok := call.Args[knowledge.Arg("time.Sleep.d")].(*ast.BasicLit)
		if !ok {
			return
		}
		n, err := strconv.Atoi(lit.Value)
		if err != nil {
			return
		}
		if n == 0 || n > 120 {
			// time.Sleep(0) is a seldom used pattern in concurrency
			// tests. >120 might be intentional. 120 was chosen
			// because the user could've meant 2 minutes.
			return
		}

		report.Report(pass, lit,
			fmt.Sprintf("sleeping for %d nanoseconds is probably a bug; be explicit if it isn't", n), report.Fixes(
				edit.Fix("explicitly use nanoseconds", edit.ReplaceWithPattern(pass, checkTimeSleepConstantPatternRns, pattern.State{"duration": lit}, lit)),
				edit.Fix("use seconds", edit.ReplaceWithPattern(pass, checkTimeSleepConstantPatternRs, pattern.State{"duration": lit}, lit))))
	}
	code.Preorder(pass, fn, (*ast.CallExpr)(nil))
	return nil, nil
}

var checkWaitgroupAddQ = pattern.MustParse(`
	(GoStmt
		(CallExpr
			(FuncLit
				_
				call@(CallExpr (Function "(*sync.WaitGroup).Add") _):_) _))`)

func CheckWaitgroupAdd(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		if m, ok := code.Match(pass, checkWaitgroupAddQ, node); ok {
			call := m.State["call"].(ast.Node)
			report.Report(pass, call, fmt.Sprintf("should call %s before starting the goroutine to avoid a race", report.Render(pass, call)))
		}
	}
	code.Preorder(pass, fn, (*ast.GoStmt)(nil))
	return nil, nil
}

func CheckInfiniteEmptyLoop(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		loop := node.(*ast.ForStmt)
		if len(loop.Body.List) != 0 || loop.Post != nil {
			return
		}

		if loop.Init != nil {
			// TODO(dh): this isn't strictly necessary, it just makes
			// the check easier.
			return
		}
		// An empty loop is bad news in two cases: 1) The loop has no
		// condition. In that case, it's just a loop that spins
		// forever and as fast as it can, keeping a core busy. 2) The
		// loop condition only consists of variable or field reads and
		// operators on those. The only way those could change their
		// value is with unsynchronised access, which constitutes a
		// data race.
		//
		// If the condition contains any function calls, its behaviour
		// is dynamic and the loop might terminate. Similarly for
		// channel receives.

		if loop.Cond != nil {
			if code.MayHaveSideEffects(pass, loop.Cond, nil) {
				return
			}
			if ident, ok := loop.Cond.(*ast.Ident); ok {
				if k, ok := pass.TypesInfo.ObjectOf(ident).(*types.Const); ok {
					if !constant.BoolVal(k.Val()) {
						// don't flag `for false {}` loops. They're a debug aid.
						return
					}
				}
			}
			report.Report(pass, loop, "loop condition never changes or has a race condition")
		}
		report.Report(pass, loop, "this loop will spin, using 100% CPU", report.ShortRange())
	}
	code.Preorder(pass, fn, (*ast.ForStmt)(nil))
	return nil, nil
}

func CheckDeferInInfiniteLoop(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		mightExit := false
		var defers []ast.Stmt
		loop := node.(*ast.ForStmt)
		if loop.Cond != nil {
			return
		}
		fn2 := func(node ast.Node) bool {
			switch stmt := node.(type) {
			case *ast.ReturnStmt:
				mightExit = true
				return false
			case *ast.BranchStmt:
				// TODO(dominikh): if this sees a break in a switch or
				// select, it doesn't check if it breaks the loop or
				// just the select/switch. This causes some false
				// negatives.
				if stmt.Tok == token.BREAK {
					mightExit = true
					return false
				}
			case *ast.DeferStmt:
				defers = append(defers, stmt)
			case *ast.FuncLit:
				// Don't look into function bodies
				return false
			}
			return true
		}
		ast.Inspect(loop.Body, fn2)
		if mightExit {
			return
		}
		for _, stmt := range defers {
			report.Report(pass, stmt, "defers in this infinite loop will never run")
		}
	}
	code.Preorder(pass, fn, (*ast.ForStmt)(nil))
	return nil, nil
}

func CheckDubiousDeferInChannelRangeLoop(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		loop := node.(*ast.RangeStmt)
		typ := pass.TypesInfo.TypeOf(loop.X)
		_, ok := typ.Underlying().(*types.Chan)
		if !ok {
			return
		}
		fn2 := func(node ast.Node) bool {
			switch stmt := node.(type) {
			case *ast.DeferStmt:
				report.Report(pass, stmt, "defers in this range loop won't run unless the channel gets closed")
			case *ast.FuncLit:
				// Don't look into function bodies
				return false
			}
			return true
		}
		ast.Inspect(loop.Body, fn2)
	}
	code.Preorder(pass, fn, (*ast.RangeStmt)(nil))
	return nil, nil
}

func CheckTestMainExit(pass *analysis.Pass) (interface{}, error) {
	if code.IsGoVersion(pass, 15) {
		// Beginning with Go 1.15, the test framework will call
		// os.Exit for us.
		return nil, nil
	}

	var (
		fnmain    ast.Node
		callsExit bool
		callsRun  bool
		arg       types.Object
	)
	fn := func(node ast.Node, push bool) bool {
		if !push {
			if fnmain != nil && node == fnmain {
				if !callsExit && callsRun {
					report.Report(pass, fnmain, "TestMain should call os.Exit to set exit code")
				}
				fnmain = nil
				callsExit = false
				callsRun = false
				arg = nil
			}
			return true
		}

		switch node := node.(type) {
		case *ast.FuncDecl:
			if fnmain != nil {
				return true
			}
			if !isTestMain(pass, node) {
				return false
			}
			fnmain = node
			arg = pass.TypesInfo.ObjectOf(node.Type.Params.List[0].Names[0])
			return true
		case *ast.CallExpr:
			if code.IsCallTo(pass, node, "os.Exit") {
				callsExit = true
				return false
			}
			sel, ok := node.Fun.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			ident, ok := sel.X.(*ast.Ident)
			if !ok {
				return true
			}
			if arg != pass.TypesInfo.ObjectOf(ident) {
				return true
			}
			if sel.Sel.Name == "Run" {
				callsRun = true
				return false
			}
			return true
		default:
			lint.ExhaustiveTypeSwitch(node)
			return true
		}
	}
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Nodes([]ast.Node{(*ast.FuncDecl)(nil), (*ast.CallExpr)(nil)}, fn)
	return nil, nil
}

func isTestMain(pass *analysis.Pass, decl *ast.FuncDecl) bool {
	if decl.Name.Name != "TestMain" {
		return false
	}
	if len(decl.Type.Params.List) != 1 {
		return false
	}
	arg := decl.Type.Params.List[0]
	if len(arg.Names) != 1 {
		return false
	}
	return code.IsOfType(pass, arg.Type, "*testing.M")
}

func CheckExec(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		call := node.(*ast.CallExpr)
		if !code.IsCallTo(pass, call, "os/exec.Command") {
			return
		}
		val, ok := code.ExprToString(pass, call.Args[knowledge.Arg("os/exec.Command.name")])
		if !ok {
			return
		}
		if !strings.Contains(val, " ") || strings.Contains(val, `\`) || strings.Contains(val, "/") {
			return
		}
		report.Report(pass, call.Args[knowledge.Arg("os/exec.Command.name")],
			"first argument to exec.Command looks like a shell command, but a program name or path are expected")
	}
	code.Preorder(pass, fn, (*ast.CallExpr)(nil))
	return nil, nil
}

func CheckLoopEmptyDefault(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		loop := node.(*ast.ForStmt)
		if len(loop.Body.List) != 1 || loop.Cond != nil || loop.Init != nil {
			return
		}
		sel, ok := loop.Body.List[0].(*ast.SelectStmt)
		if !ok {
			return
		}
		for _, c := range sel.Body.List {
			// FIXME this leaves behind an empty line, and possibly
			// comments in the default branch. We can't easily fix
			// either.
			if comm, ok := c.(*ast.CommClause); ok && comm.Comm == nil && len(comm.Body) == 0 {
				report.Report(pass, comm, "should not have an empty default case in a for+select loop; the loop will spin",
					report.Fixes(edit.Fix("remove empty default branch", edit.Delete(comm))))
				// there can only be one default case
				break
			}
		}
	}
	code.Preorder(pass, fn, (*ast.ForStmt)(nil))
	return nil, nil
}

func CheckLhsRhsIdentical(pass *analysis.Pass) (interface{}, error) {
	var isFloat func(T types.Type) bool
	isFloat = func(T types.Type) bool {
		switch T := T.Underlying().(type) {
		case *types.Basic:
			kind := T.Kind()
			return kind == types.Float32 || kind == types.Float64
		case *types.Array:
			return isFloat(T.Elem())
		case *types.Struct:
			for i := 0; i < T.NumFields(); i++ {
				if !isFloat(T.Field(i).Type()) {
					return false
				}
			}
			return true
		default:
			return false
		}
	}

	// TODO(dh): this check ignores the existence of side-effects and
	// happily flags fn() == fn() – so far, we've had nobody complain
	// about a false positive, and it's caught several bugs in real
	// code.
	//
	// We special case functions from the math/rand package. Someone ran
	// into the following false positive: "rand.Intn(2) - rand.Intn(2), which I wrote to generate values {-1, 0, 1} with {0.25, 0.5, 0.25} probability."
	fn := func(node ast.Node) {
		op := node.(*ast.BinaryExpr)
		switch op.Op {
		case token.EQL, token.NEQ:
		case token.SUB, token.QUO, token.AND, token.REM, token.OR, token.XOR, token.AND_NOT,
			token.LAND, token.LOR, token.LSS, token.GTR, token.LEQ, token.GEQ:
		default:
			// For some ops, such as + and *, it can make sense to
			// have identical operands
			return
		}

		if isFloat(pass.TypesInfo.TypeOf(op.X)) {
			// 'float <op> float' makes sense for several operators.
			// We've tried keeping an exact list of operators to allow, but floats keep surprising us. Let's just give up instead.
			return
		}

		if reflect.TypeOf(op.X) != reflect.TypeOf(op.Y) {
			return
		}
		if report.Render(pass, op.X) != report.Render(pass, op.Y) {
			return
		}
		l1, ok1 := op.X.(*ast.BasicLit)
		l2, ok2 := op.Y.(*ast.BasicLit)
		if ok1 && ok2 && l1.Kind == token.INT && l2.Kind == l1.Kind && l1.Value == "0" && l2.Value == l1.Value && code.IsGenerated(pass, l1.Pos()) {
			// cgo generates the following function call:
			// _cgoCheckPointer(_cgoBase0, 0 == 0) – it uses 0 == 0
			// instead of true in case the user shadowed the
			// identifier. Ideally we'd restrict this exception to
			// calls of _cgoCheckPointer, but it's not worth the
			// hassle of keeping track of the stack. <lit> <op> <lit>
			// are very rare to begin with, and we're mostly checking
			// for them to catch typos such as 1 == 1 where the user
			// meant to type i == 1. The odds of a false negative for
			// 0 == 0 are slim.
			return
		}

		if expr, ok := op.X.(*ast.CallExpr); ok {
			call := code.CallName(pass, expr)
			switch call {
			case "math/rand.Int",
				"math/rand.Int31",
				"math/rand.Int31n",
				"math/rand.Int63",
				"math/rand.Int63n",
				"math/rand.Intn",
				"math/rand.Uint32",
				"math/rand.Uint64",
				"math/rand.ExpFloat64",
				"math/rand.Float32",
				"math/rand.Float64",
				"math/rand.NormFloat64",
				"(*math/rand.Rand).Int",
				"(*math/rand.Rand).Int31",
				"(*math/rand.Rand).Int31n",
				"(*math/rand.Rand).Int63",
				"(*math/rand.Rand).Int63n",
				"(*math/rand.Rand).Intn",
				"(*math/rand.Rand).Uint32",
				"(*math/rand.Rand).Uint64",
				"(*math/rand.Rand).ExpFloat64",
				"(*math/rand.Rand).Float32",
				"(*math/rand.Rand).Float64",
				"(*math/rand.Rand).NormFloat64":
				return
			}
		}

		report.Report(pass, op, fmt.Sprintf("identical expressions on the left and right side of the '%s' operator", op.Op))
	}
	code.Preorder(pass, fn, (*ast.BinaryExpr)(nil))
	return nil, nil
}

func CheckScopedBreak(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		var body *ast.BlockStmt
		switch node := node.(type) {
		case *ast.ForStmt:
			body = node.Body
		case *ast.RangeStmt:
			body = node.Body
		default:
			lint.ExhaustiveTypeSwitch(node)
		}
		for _, stmt := range body.List {
			var blocks [][]ast.Stmt
			switch stmt := stmt.(type) {
			case *ast.SwitchStmt:
				for _, c := range stmt.Body.List {
					blocks = append(blocks, c.(*ast.CaseClause).Body)
				}
			case *ast.SelectStmt:
				for _, c := range stmt.Body.List {
					blocks = append(blocks, c.(*ast.CommClause).Body)
				}
			default:
				continue
			}

			for _, body := range blocks {
				if len(body) == 0 {
					continue
				}
				lasts := []ast.Stmt{body[len(body)-1]}
				// TODO(dh): unfold all levels of nested block
				// statements, not just a single level if statement
				if ifs, ok := lasts[0].(*ast.IfStmt); ok {
					if len(ifs.Body.List) == 0 {
						continue
					}
					lasts[0] = ifs.Body.List[len(ifs.Body.List)-1]

					if block, ok := ifs.Else.(*ast.BlockStmt); ok {
						if len(block.List) != 0 {
							lasts = append(lasts, block.List[len(block.List)-1])
						}
					}
				}
				for _, last := range lasts {
					branch, ok := last.(*ast.BranchStmt)
					if !ok || branch.Tok != token.BREAK || branch.Label != nil {
						continue
					}
					report.Report(pass, branch, "ineffective break statement. Did you mean to break out of the outer loop?")
				}
			}
		}
	}
	code.Preorder(pass, fn, (*ast.ForStmt)(nil), (*ast.RangeStmt)(nil))
	return nil, nil
}

func CheckUnsafePrintf(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		call := node.(*ast.CallExpr)
		name := code.CallName(pass, call)
		var arg int

		switch name {
		case "fmt.Printf", "fmt.Sprintf", "log.Printf":
			arg = knowledge.Arg("fmt.Printf.format")
		case "fmt.Fprintf":
			arg = knowledge.Arg("fmt.Fprintf.format")
		default:
			return
		}
		if len(call.Args) != arg+1 {
			return
		}
		switch call.Args[arg].(type) {
		case *ast.CallExpr, *ast.Ident:
		default:
			return
		}

		if _, ok := pass.TypesInfo.TypeOf(call.Args[arg]).(*types.Tuple); ok {
			// the called function returns multiple values and got
			// splatted into the call. for all we know, it is
			// returning good arguments.
			return
		}

		alt := name[:len(name)-1]
		report.Report(pass, call,
			"printf-style function with dynamic format string and no further arguments should use print-style function instead",
			report.Fixes(edit.Fix(fmt.Sprintf("use %s instead of %s", alt, name), edit.ReplaceWithString(pass.Fset, call.Fun, alt))))
	}
	code.Preorder(pass, fn, (*ast.CallExpr)(nil))
	return nil, nil
}

func CheckEarlyDefer(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		block := node.(*ast.BlockStmt)
		if len(block.List) < 2 {
			return
		}
		for i, stmt := range block.List {
			if i == len(block.List)-1 {
				break
			}
			assign, ok := stmt.(*ast.AssignStmt)
			if !ok {
				continue
			}
			if len(assign.Rhs) != 1 {
				continue
			}
			if len(assign.Lhs) < 2 {
				continue
			}
			if lhs, ok := assign.Lhs[len(assign.Lhs)-1].(*ast.Ident); ok && lhs.Name == "_" {
				continue
			}
			call, ok := assign.Rhs[0].(*ast.CallExpr)
			if !ok {
				continue
			}
			sig, ok := pass.TypesInfo.TypeOf(call.Fun).(*types.Signature)
			if !ok {
				continue
			}
			if sig.Results().Len() < 2 {
				continue
			}
			last := sig.Results().At(sig.Results().Len() - 1)
			// FIXME(dh): check that it's error from universe, not
			// another type of the same name
			if last.Type().String() != "error" {
				continue
			}
			lhs, ok := assign.Lhs[0].(*ast.Ident)
			if !ok {
				continue
			}
			def, ok := block.List[i+1].(*ast.DeferStmt)
			if !ok {
				continue
			}
			sel, ok := def.Call.Fun.(*ast.SelectorExpr)
			if !ok {
				continue
			}
			ident, ok := selectorX(sel).(*ast.Ident)
			if !ok {
				continue
			}
			if ident.Obj != lhs.Obj {
				continue
			}
			if sel.Sel.Name != "Close" {
				continue
			}
			report.Report(pass, def, fmt.Sprintf("should check returned error before deferring %s", report.Render(pass, def.Call)))
		}
	}
	code.Preorder(pass, fn, (*ast.BlockStmt)(nil))
	return nil, nil
}

func selectorX(sel *ast.SelectorExpr) ast.Node {
	switch x := sel.X.(type) {
	case *ast.SelectorExpr:
		return selectorX(x)
	default:
		return x
	}
}

func CheckEmptyCriticalSection(pass *analysis.Pass) (interface{}, error) {
	if pass.Pkg.Path() == "sync_test" {
		// exception for the sync package's tests
		return nil, nil
	}

	// Initially it might seem like this check would be easier to
	// implement using IR. After all, we're only checking for two
	// consecutive method calls. In reality, however, there may be any
	// number of other instructions between the lock and unlock, while
	// still constituting an empty critical section. For example,
	// given `m.x().Lock(); m.x().Unlock()`, there will be a call to
	// x(). In the AST-based approach, this has a tiny potential for a
	// false positive (the second call to x might be doing work that
	// is protected by the mutex). In an IR-based approach, however,
	// it would miss a lot of real bugs.

	mutexParams := func(s ast.Stmt) (x ast.Expr, funcName string, ok bool) {
		expr, ok := s.(*ast.ExprStmt)
		if !ok {
			return nil, "", false
		}
		call, ok := expr.X.(*ast.CallExpr)
		if !ok {
			return nil, "", false
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return nil, "", false
		}

		fn, ok := pass.TypesInfo.ObjectOf(sel.Sel).(*types.Func)
		if !ok {
			return nil, "", false
		}
		sig := fn.Type().(*types.Signature)
		if sig.Params().Len() != 0 || sig.Results().Len() != 0 {
			return nil, "", false
		}

		return sel.X, fn.Name(), true
	}

	fn := func(node ast.Node) {
		block := node.(*ast.BlockStmt)
		if len(block.List) < 2 {
			return
		}
		for i := range block.List[:len(block.List)-1] {
			sel1, method1, ok1 := mutexParams(block.List[i])
			sel2, method2, ok2 := mutexParams(block.List[i+1])

			if !ok1 || !ok2 || report.Render(pass, sel1) != report.Render(pass, sel2) {
				continue
			}
			if (method1 == "Lock" && method2 == "Unlock") ||
				(method1 == "RLock" && method2 == "RUnlock") {
				report.Report(pass, block.List[i+1], "empty critical section")
			}
		}
	}
	code.Preorder(pass, fn, (*ast.BlockStmt)(nil))
	return nil, nil
}

var (
	// cgo produces code like fn(&*_Cvar_kSomeCallbacks) which we don't
	// want to flag.
	cgoIdent               = regexp.MustCompile(`^_C(func|var)_.+$`)
	checkIneffectiveCopyQ1 = pattern.MustParse(`(UnaryExpr "&" (StarExpr obj))`)
	checkIneffectiveCopyQ2 = pattern.MustParse(`(StarExpr (UnaryExpr "&" _))`)
)

func CheckIneffectiveCopy(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		if m, ok := code.Match(pass, checkIneffectiveCopyQ1, node); ok {
			if ident, ok := m.State["obj"].(*ast.Ident); !ok || !cgoIdent.MatchString(ident.Name) {
				report.Report(pass, node, "&*x will be simplified to x. It will not copy x.")
			}
		} else if _, ok := code.Match(pass, checkIneffectiveCopyQ2, node); ok {
			report.Report(pass, node, "*&x will be simplified to x. It will not copy x.")
		}
	}
	code.Preorder(pass, fn, (*ast.UnaryExpr)(nil), (*ast.StarExpr)(nil))
	return nil, nil
}

func CheckCanonicalHeaderKey(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node, push bool) bool {
		if !push {
			return false
		}
		if assign, ok := node.(*ast.AssignStmt); ok {
			// TODO(dh): This risks missing some Header reads, for
			// example in `h1["foo"] = h2["foo"]` – these edge
			// cases are probably rare enough to ignore for now.
			for _, expr := range assign.Lhs {
				op, ok := expr.(*ast.IndexExpr)
				if !ok {
					continue
				}
				if code.IsOfType(pass, op.X, "net/http.Header") {
					return false
				}
			}
			return true
		}
		op, ok := node.(*ast.IndexExpr)
		if !ok {
			return true
		}
		if !code.IsOfType(pass, op.X, "net/http.Header") {
			return true
		}
		s, ok := code.ExprToString(pass, op.Index)
		if !ok {
			return true
		}
		canonical := http.CanonicalHeaderKey(s)
		if s == canonical {
			return true
		}
		var fix analysis.SuggestedFix
		switch op.Index.(type) {
		case *ast.BasicLit:
			fix = edit.Fix("canonicalize header key", edit.ReplaceWithString(pass.Fset, op.Index, strconv.Quote(canonical)))
		case *ast.Ident:
			call := &ast.CallExpr{
				Fun:  edit.Selector("http", "CanonicalHeaderKey"),
				Args: []ast.Expr{op.Index},
			}
			fix = edit.Fix("wrap in http.CanonicalHeaderKey", edit.ReplaceWithNode(pass.Fset, op.Index, call))
		}
		msg := fmt.Sprintf("keys in http.Header are canonicalized, %q is not canonical; fix the constant or use http.CanonicalHeaderKey", s)
		if fix.Message != "" {
			report.Report(pass, op, msg, report.Fixes(fix))
		} else {
			report.Report(pass, op, msg)
		}
		return true
	}
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Nodes([]ast.Node{(*ast.AssignStmt)(nil), (*ast.IndexExpr)(nil)}, fn)
	return nil, nil
}

func CheckBenchmarkN(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		assign := node.(*ast.AssignStmt)
		if len(assign.Lhs) != 1 || len(assign.Rhs) != 1 {
			return
		}
		sel, ok := assign.Lhs[0].(*ast.SelectorExpr)
		if !ok {
			return
		}
		if sel.Sel.Name != "N" {
			return
		}
		if !code.IsOfType(pass, sel.X, "*testing.B") {
			return
		}
		report.Report(pass, assign, fmt.Sprintf("should not assign to %s", report.Render(pass, sel)))
	}
	code.Preorder(pass, fn, (*ast.AssignStmt)(nil))
	return nil, nil
}

func CheckUnreadVariableValues(pass *analysis.Pass) (interface{}, error) {
	for _, fn := range pass.ResultOf[buildir.Analyzer].(*buildir.IR).SrcFuncs {
		if irutil.IsExample(fn) {
			continue
		}
		node := fn.Source()
		if node == nil {
			continue
		}
		if gen, ok := code.Generator(pass, node.Pos()); ok && gen == facts.Goyacc {
			// Don't flag unused values in code generated by goyacc.
			// There may be hundreds of those due to the way the state
			// machine is constructed.
			continue
		}

		switchTags := map[ir.Value]struct{}{}
		ast.Inspect(node, func(node ast.Node) bool {
			s, ok := node.(*ast.SwitchStmt)
			if !ok {
				return true
			}
			v, _ := fn.ValueForExpr(s.Tag)
			switchTags[v] = struct{}{}
			return true
		})

		// OPT(dh): don't use a map, possibly use a bitset
		var hasUse func(v ir.Value, seen map[ir.Value]struct{}) bool
		hasUse = func(v ir.Value, seen map[ir.Value]struct{}) bool {
			if _, ok := seen[v]; ok {
				return false
			}
			if _, ok := switchTags[v]; ok {
				return true
			}
			refs := v.Referrers()
			if refs == nil {
				// TODO investigate why refs can be nil
				return true
			}
			for _, ref := range *refs {
				switch ref := ref.(type) {
				case *ir.DebugRef:
				case *ir.Sigma:
					if seen == nil {
						seen = map[ir.Value]struct{}{}
					}
					seen[v] = struct{}{}
					if hasUse(ref, seen) {
						return true
					}
				case *ir.Phi:
					if seen == nil {
						seen = map[ir.Value]struct{}{}
					}
					seen[v] = struct{}{}
					if hasUse(ref, seen) {
						return true
					}
				default:
					return true
				}
			}
			return false
		}

		ast.Inspect(node, func(node ast.Node) bool {
			assign, ok := node.(*ast.AssignStmt)
			if !ok {
				return true
			}
			if len(assign.Lhs) > 1 && len(assign.Rhs) == 1 {
				// Either a function call with multiple return values,
				// or a comma-ok assignment

				val, _ := fn.ValueForExpr(assign.Rhs[0])
				if val == nil {
					return true
				}
				refs := val.Referrers()
				if refs == nil {
					return true
				}
				for _, ref := range *refs {
					ex, ok := ref.(*ir.Extract)
					if !ok {
						continue
					}
					if !hasUse(ex, nil) {
						lhs := assign.Lhs[ex.Index]
						if ident, ok := lhs.(*ast.Ident); !ok || ok && ident.Name == "_" {
							continue
						}
						report.Report(pass, assign, fmt.Sprintf("this value of %s is never used", lhs))
					}
				}
				return true
			}
			for i, lhs := range assign.Lhs {
				rhs := assign.Rhs[i]
				if ident, ok := lhs.(*ast.Ident); !ok || ok && ident.Name == "_" {
					continue
				}
				val, _ := fn.ValueForExpr(rhs)
				if val == nil {
					continue
				}

				if _, ok := val.(*ir.Const); ok {
					// a zero-valued constant, for example in 'foo := []string(nil)'
					continue
				}
				if !hasUse(val, nil) {
					report.Report(pass, assign, fmt.Sprintf("this value of %s is never used", lhs))
				}
			}
			return true
		})
	}
	return nil, nil
}

func CheckPredeterminedBooleanExprs(pass *analysis.Pass) (interface{}, error) {
	for _, fn := range pass.ResultOf[buildir.Analyzer].(*buildir.IR).SrcFuncs {
		for _, block := range fn.Blocks {
			for _, ins := range block.Instrs {
				binop, ok := ins.(*ir.BinOp)
				if !ok {
					continue
				}
				switch binop.Op {
				case token.GTR, token.LSS, token.EQL, token.NEQ, token.LEQ, token.GEQ:
				default:
					continue
				}

				xs, ok1 := consts(binop.X, nil, nil)
				ys, ok2 := consts(binop.Y, nil, nil)
				if !ok1 || !ok2 || len(xs) == 0 || len(ys) == 0 {
					continue
				}

				trues := 0
				for _, x := range xs {
					for _, y := range ys {
						if x.Value == nil {
							if y.Value == nil {
								trues++
							}
							continue
						}
						if constant.Compare(x.Value, binop.Op, y.Value) {
							trues++
						}
					}
				}
				b := trues != 0
				if trues == 0 || trues == len(xs)*len(ys) {
					report.Report(pass, binop, fmt.Sprintf("binary expression is always %t for all possible values (%s %s %s)", b, xs, binop.Op, ys))
				}
			}
		}
	}
	return nil, nil
}

func CheckNilMaps(pass *analysis.Pass) (interface{}, error) {
	for _, fn := range pass.ResultOf[buildir.Analyzer].(*buildir.IR).SrcFuncs {
		for _, block := range fn.Blocks {
			for _, ins := range block.Instrs {
				mu, ok := ins.(*ir.MapUpdate)
				if !ok {
					continue
				}
				c, ok := irutil.Flatten(mu.Map).(*ir.Const)
				if !ok {
					continue
				}
				if c.Value != nil {
					continue
				}
				report.Report(pass, mu, "assignment to nil map")
			}
		}
	}
	return nil, nil
}

func CheckExtremeComparison(pass *analysis.Pass) (interface{}, error) {
	isobj := func(expr ast.Expr, name string) bool {
		sel, ok := expr.(*ast.SelectorExpr)
		if !ok {
			return false
		}
		return typeutil.IsObject(pass.TypesInfo.ObjectOf(sel.Sel), name)
	}

	fn := func(node ast.Node) {
		expr := node.(*ast.BinaryExpr)
		tx := pass.TypesInfo.TypeOf(expr.X)
		basic, ok := tx.Underlying().(*types.Basic)
		if !ok {
			return
		}

		var max string
		var min string

		switch basic.Kind() {
		case types.Uint8:
			max = "math.MaxUint8"
		case types.Uint16:
			max = "math.MaxUint16"
		case types.Uint32:
			max = "math.MaxUint32"
		case types.Uint64:
			max = "math.MaxUint64"
		case types.Uint:
			max = "math.MaxUint64"

		case types.Int8:
			min = "math.MinInt8"
			max = "math.MaxInt8"
		case types.Int16:
			min = "math.MinInt16"
			max = "math.MaxInt16"
		case types.Int32:
			min = "math.MinInt32"
			max = "math.MaxInt32"
		case types.Int64:
			min = "math.MinInt64"
			max = "math.MaxInt64"
		case types.Int:
			min = "math.MinInt64"
			max = "math.MaxInt64"
		}

		if (expr.Op == token.GTR || expr.Op == token.GEQ) && isobj(expr.Y, max) ||
			(expr.Op == token.LSS || expr.Op == token.LEQ) && isobj(expr.X, max) {
			report.Report(pass, expr, fmt.Sprintf("no value of type %s is greater than %s", basic, max))
		}
		if expr.Op == token.LEQ && isobj(expr.Y, max) ||
			expr.Op == token.GEQ && isobj(expr.X, max) {
			report.Report(pass, expr, fmt.Sprintf("every value of type %s is <= %s", basic, max))
		}

		if (basic.Info() & types.IsUnsigned) != 0 {
			if (expr.Op == token.LSS && astutil.IsIntLiteral(expr.Y, "0")) ||
				(expr.Op == token.GTR && astutil.IsIntLiteral(expr.X, "0")) {
				report.Report(pass, expr, fmt.Sprintf("no value of type %s is less than 0", basic))
			}
			if expr.Op == token.GEQ && astutil.IsIntLiteral(expr.Y, "0") ||
				expr.Op == token.LEQ && astutil.IsIntLiteral(expr.X, "0") {
				report.Report(pass, expr, fmt.Sprintf("every value of type %s is >= 0", basic))
			}
		} else {
			if (expr.Op == token.LSS || expr.Op == token.LEQ) && isobj(expr.Y, min) ||
				(expr.Op == token.GTR || expr.Op == token.GEQ) && isobj(expr.X, min) {
				report.Report(pass, expr, fmt.Sprintf("no value of type %s is less than %s", basic, min))
			}
			if expr.Op == token.GEQ && isobj(expr.Y, min) ||
				expr.Op == token.LEQ && isobj(expr.X, min) {
				report.Report(pass, expr, fmt.Sprintf("every value of type %s is >= %s", basic, min))
			}
		}

	}
	code.Preorder(pass, fn, (*ast.BinaryExpr)(nil))
	return nil, nil
}

func consts(val ir.Value, out []*ir.Const, visitedPhis map[string]bool) ([]*ir.Const, bool) {
	if visitedPhis == nil {
		visitedPhis = map[string]bool{}
	}
	var ok bool
	switch val := val.(type) {
	case *ir.Phi:
		if visitedPhis[val.Name()] {
			break
		}
		visitedPhis[val.Name()] = true
		vals := val.Operands(nil)
		for _, phival := range vals {
			out, ok = consts(*phival, out, visitedPhis)
			if !ok {
				return nil, false
			}
		}
	case *ir.Const:
		out = append(out, val)
	case *ir.Convert:
		out, ok = consts(val.X, out, visitedPhis)
		if !ok {
			return nil, false
		}
	default:
		return nil, false
	}
	if len(out) < 2 {
		return out, true
	}
	uniq := []*ir.Const{out[0]}
	for _, val := range out[1:] {
		if val.Value == uniq[len(uniq)-1].Value {
			continue
		}
		uniq = append(uniq, val)
	}
	return uniq, true
}

func CheckLoopCondition(pass *analysis.Pass) (interface{}, error) {
	for _, fn := range pass.ResultOf[buildir.Analyzer].(*buildir.IR).SrcFuncs {
		cb := func(node ast.Node) bool {
			loop, ok := node.(*ast.ForStmt)
			if !ok {
				return true
			}
			if loop.Init == nil || loop.Cond == nil || loop.Post == nil {
				return true
			}
			init, ok := loop.Init.(*ast.AssignStmt)
			if !ok || len(init.Lhs) != 1 || len(init.Rhs) != 1 {
				return true
			}
			cond, ok := loop.Cond.(*ast.BinaryExpr)
			if !ok {
				return true
			}
			x, ok := cond.X.(*ast.Ident)
			if !ok {
				return true
			}
			lhs, ok := init.Lhs[0].(*ast.Ident)
			if !ok {
				return true
			}
			if x.Obj != lhs.Obj {
				return true
			}
			if _, ok := loop.Post.(*ast.IncDecStmt); !ok {
				return true
			}

			v, isAddr := fn.ValueForExpr(cond.X)
			if v == nil || isAddr {
				return true
			}
			switch v := v.(type) {
			case *ir.Phi:
				ops := v.Operands(nil)
				if len(ops) != 2 {
					return true
				}
				_, ok := (*ops[0]).(*ir.Const)
				if !ok {
					return true
				}
				sigma, ok := (*ops[1]).(*ir.Sigma)
				if !ok {
					return true
				}
				if sigma.X != v {
					return true
				}
			case *ir.Load:
				return true
			}
			report.Report(pass, cond, "variable in loop condition never changes")

			return true
		}
		if source := fn.Source(); source != nil {
			ast.Inspect(source, cb)
		}
	}
	return nil, nil
}

func CheckArgOverwritten(pass *analysis.Pass) (interface{}, error) {
	for _, fn := range pass.ResultOf[buildir.Analyzer].(*buildir.IR).SrcFuncs {
		cb := func(node ast.Node) bool {
			var typ *ast.FuncType
			var body *ast.BlockStmt
			switch fn := node.(type) {
			case *ast.FuncDecl:
				typ = fn.Type
				body = fn.Body
			case *ast.FuncLit:
				typ = fn.Type
				body = fn.Body
			}
			if body == nil {
				return true
			}
			if len(typ.Params.List) == 0 {
				return true
			}
			for _, field := range typ.Params.List {
				for _, arg := range field.Names {
					obj := pass.TypesInfo.ObjectOf(arg)
					var irobj *ir.Parameter
					for _, param := range fn.Params {
						if param.Object() == obj {
							irobj = param
							break
						}
					}
					if irobj == nil {
						continue
					}
					refs := irobj.Referrers()
					if refs == nil {
						continue
					}
					if len(irutil.FilterDebug(*refs)) != 0 {
						continue
					}

					var assignment ast.Node
					ast.Inspect(body, func(node ast.Node) bool {
						if assignment != nil {
							return false
						}
						assign, ok := node.(*ast.AssignStmt)
						if !ok {
							return true
						}
						for _, lhs := range assign.Lhs {
							ident, ok := lhs.(*ast.Ident)
							if !ok {
								continue
							}
							if pass.TypesInfo.ObjectOf(ident) == obj {
								assignment = assign
								return false
							}
						}
						return true
					})
					if assignment != nil {
						report.Report(pass, arg, fmt.Sprintf("argument %s is overwritten before first use", arg),
							report.Related(assignment, fmt.Sprintf("assignment to %s", arg)))
					}
				}
			}
			return true
		}
		if source := fn.Source(); source != nil {
			ast.Inspect(source, cb)
		}
	}
	return nil, nil
}

func CheckIneffectiveLoop(pass *analysis.Pass) (interface{}, error) {
	// This check detects some, but not all unconditional loop exits.
	// We give up in the following cases:
	//
	// - a goto anywhere in the loop. The goto might skip over our
	// return, and we don't check that it doesn't.
	//
	// - any nested, unlabelled continue, even if it is in another
	// loop or closure.
	fn := func(node ast.Node) {
		var body *ast.BlockStmt
		switch fn := node.(type) {
		case *ast.FuncDecl:
			body = fn.Body
		case *ast.FuncLit:
			body = fn.Body
		default:
			lint.ExhaustiveTypeSwitch(node)
		}
		if body == nil {
			return
		}
		labels := map[*ast.Object]ast.Stmt{}
		ast.Inspect(body, func(node ast.Node) bool {
			label, ok := node.(*ast.LabeledStmt)
			if !ok {
				return true
			}
			labels[label.Label.Obj] = label.Stmt
			return true
		})

		ast.Inspect(body, func(node ast.Node) bool {
			var loop ast.Node
			var body *ast.BlockStmt
			switch node := node.(type) {
			case *ast.ForStmt:
				body = node.Body
				loop = node
			case *ast.RangeStmt:
				typ := pass.TypesInfo.TypeOf(node.X)
				if _, ok := typ.Underlying().(*types.Map); ok {
					// looping once over a map is a valid pattern for
					// getting an arbitrary element.
					return true
				}
				body = node.Body
				loop = node
			default:
				return true
			}
			if len(body.List) < 2 {
				// TODO(dh): is this check needed? when body.List < 2,
				// then we can't find both an unconditional exit and a
				// branching statement (if, ...). and we don't flag
				// unconditional exits if there has been no branching
				// in the loop body.

				// avoid flagging the somewhat common pattern of using
				// a range loop to get the first element in a slice,
				// or the first rune in a string.
				return true
			}
			var unconditionalExit ast.Node
			hasBranching := false
			for _, stmt := range body.List {
				switch stmt := stmt.(type) {
				case *ast.BranchStmt:
					switch stmt.Tok {
					case token.BREAK:
						if stmt.Label == nil || labels[stmt.Label.Obj] == loop {
							unconditionalExit = stmt
						}
					case token.CONTINUE:
						if stmt.Label == nil || labels[stmt.Label.Obj] == loop {
							unconditionalExit = nil
							return false
						}
					}
				case *ast.ReturnStmt:
					unconditionalExit = stmt
				case *ast.IfStmt, *ast.ForStmt, *ast.RangeStmt, *ast.SwitchStmt, *ast.SelectStmt:
					hasBranching = true
				}
			}
			if unconditionalExit == nil || !hasBranching {
				return false
			}
			ast.Inspect(body, func(node ast.Node) bool {
				if branch, ok := node.(*ast.BranchStmt); ok {

					switch branch.Tok {
					case token.GOTO:
						unconditionalExit = nil
						return false
					case token.CONTINUE:
						if branch.Label != nil && labels[branch.Label.Obj] != loop {
							return true
						}
						unconditionalExit = nil
						return false
					}
				}
				return true
			})
			if unconditionalExit != nil {
				report.Report(pass, unconditionalExit, "the surrounding loop is unconditionally terminated")
			}
			return true
		})
	}
	code.Preorder(pass, fn, (*ast.FuncDecl)(nil), (*ast.FuncLit)(nil))
	return nil, nil
}

var checkNilContextQ = pattern.MustParse(`(CallExpr fun@(Function _) (Builtin "nil"):_)`)

func CheckNilContext(pass *analysis.Pass) (interface{}, error) {
	todo := &ast.CallExpr{
		Fun: edit.Selector("context", "TODO"),
	}
	bg := &ast.CallExpr{
		Fun: edit.Selector("context", "Background"),
	}
	fn := func(node ast.Node) {
		m, ok := code.Match(pass, checkNilContextQ, node)
		if !ok {
			return
		}

		call := node.(*ast.CallExpr)
		fun, ok := m.State["fun"].(*types.Func)
		if !ok {
			// it might also be a builtin
			return
		}
		sig := fun.Type().(*types.Signature)
		if sig.Params().Len() == 0 {
			// Our CallExpr might've matched a method expression, like
			// (*T).Foo(nil) – here, nil isn't the first argument of
			// the Foo method, but the method receiver.
			return
		}
		if !typeutil.IsType(sig.Params().At(0).Type(), "context.Context") {
			return
		}
		report.Report(pass, call.Args[0],
			"do not pass a nil Context, even if a function permits it; pass context.TODO if you are unsure about which Context to use", report.Fixes(
				edit.Fix("use context.TODO", edit.ReplaceWithNode(pass.Fset, call.Args[0], todo)),
				edit.Fix("use context.Background", edit.ReplaceWithNode(pass.Fset, call.Args[0], bg))))
	}
	code.Preorder(pass, fn, (*ast.CallExpr)(nil))
	return nil, nil
}

var (
	checkSeekerQ = pattern.MustParse(`(CallExpr fun@(SelectorExpr _ (Ident "Seek")) [arg1@(SelectorExpr (Ident "io") (Ident (Or "SeekStart" "SeekCurrent" "SeekEnd"))) arg2])`)
	checkSeekerR = pattern.MustParse(`(CallExpr fun [arg2 arg1])`)
)

func CheckSeeker(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		if _, edits, ok := code.MatchAndEdit(pass, checkSeekerQ, checkSeekerR, node); ok {
			report.Report(pass, node, "the first argument of io.Seeker is the offset, but an io.Seek* constant is being used instead",
				report.Fixes(edit.Fix("swap arguments", edits...)))
		}
	}
	code.Preorder(pass, fn, (*ast.CallExpr)(nil))
	return nil, nil
}

func CheckIneffectiveAppend(pass *analysis.Pass) (interface{}, error) {
	isAppend := func(ins ir.Value) bool {
		call, ok := ins.(*ir.Call)
		if !ok {
			return false
		}
		if call.Call.IsInvoke() {
			return false
		}
		if builtin, ok := call.Call.Value.(*ir.Builtin); !ok || builtin.Name() != "append" {
			return false
		}
		return true
	}

	// We have to be careful about aliasing.
	// Multiple slices may refer to the same backing array,
	// making appends observable even when we don't see the result of append be used anywhere.
	//
	// We will have to restrict ourselves to slices that have been allocated within the function,
	// haven't been sliced,
	// and haven't been passed anywhere that could retain them (such as function calls or memory stores).
	//
	// We check whether an append should be flagged in two steps.
	//
	// In the first step, we look at the data flow graph, starting in reverse from the argument to append, till we reach the root.
	// This graph must only consist of the following instructions:
	//
	// - phi
	// - sigma
	// - slice
	// - const nil
	// - MakeSlice
	// - Alloc
	// - calls to append
	//
	// If this step succeeds, we look at all referrers of the values found in the first step, recursively.
	// These referrers must either be in the set of values found in the first step,
	// be DebugRefs,
	// or fulfill the same type requirements as step 1, with the exception of appends, which are forbidden.
	//
	// If both steps succeed then we know that the backing array hasn't been aliased in an observable manner.
	//
	// We could relax these restrictions by making use of additional information:
	// - if we passed the slice to a function that doesn't retain the slice then we can still flag it
	// - if a slice has been sliced but is dead afterwards, we can flag appends to the new slice

	// OPT(dh): We could cache the results of both validate functions.
	// However, we only use these functions on values that we otherwise want to flag, which are very few.
	// Not caching values hasn't increased the runtimes for the standard library nor k8s.
	var validateArgument func(v ir.Value, seen map[ir.Value]struct{}) bool
	validateArgument = func(v ir.Value, seen map[ir.Value]struct{}) bool {
		if _, ok := seen[v]; ok {
			// break cycle
			return true
		}
		seen[v] = struct{}{}
		switch v := v.(type) {
		case *ir.Phi:
			for _, edge := range v.Edges {
				if !validateArgument(edge, seen) {
					return false
				}
			}
			return true
		case *ir.Sigma:
			return validateArgument(v.X, seen)
		case *ir.Slice:
			return validateArgument(v.X, seen)
		case *ir.Const:
			return true
		case *ir.MakeSlice:
			return true
		case *ir.Alloc:
			return true
		case *ir.Call:
			if isAppend(v) {
				return validateArgument(v.Call.Args[0], seen)
			}
			return false
		default:
			return false
		}
	}

	var validateReferrers func(v ir.Value, seen map[ir.Instruction]struct{}) bool
	validateReferrers = func(v ir.Value, seen map[ir.Instruction]struct{}) bool {
		for _, ref := range *v.Referrers() {
			if _, ok := seen[ref]; ok {
				continue
			}

			seen[ref] = struct{}{}
			switch ref.(type) {
			case *ir.Phi:
			case *ir.Sigma:
			case *ir.Slice:
			case *ir.Const:
			case *ir.MakeSlice:
			case *ir.Alloc:
			case *ir.DebugRef:
			default:
				return false
			}

			if ref, ok := ref.(ir.Value); ok {
				if !validateReferrers(ref, seen) {
					return false
				}
			}
		}
		return true
	}

	for _, fn := range pass.ResultOf[buildir.Analyzer].(*buildir.IR).SrcFuncs {
		for _, block := range fn.Blocks {
			for _, ins := range block.Instrs {
				val, ok := ins.(ir.Value)
				if !ok || !isAppend(val) {
					continue
				}

				isUsed := false
				visited := map[ir.Instruction]bool{}
				var walkRefs func(refs []ir.Instruction)
				walkRefs = func(refs []ir.Instruction) {
				loop:
					for _, ref := range refs {
						if visited[ref] {
							continue
						}
						visited[ref] = true
						if _, ok := ref.(*ir.DebugRef); ok {
							continue
						}
						switch ref := ref.(type) {
						case *ir.Phi:
							walkRefs(*ref.Referrers())
						case *ir.Sigma:
							walkRefs(*ref.Referrers())
						case ir.Value:
							if !isAppend(ref) {
								isUsed = true
							} else {
								walkRefs(*ref.Referrers())
							}
						case ir.Instruction:
							isUsed = true
							break loop
						}
					}
				}

				refs := val.Referrers()
				if refs == nil {
					continue
				}
				walkRefs(*refs)

				if isUsed {
					continue
				}

				seen := map[ir.Value]struct{}{}
				if !validateArgument(ins.(*ir.Call).Call.Args[0], seen) {
					continue
				}

				seen2 := map[ir.Instruction]struct{}{}
				for k := range seen {
					// the only values we allow are also instructions, so this type assertion cannot fail
					seen2[k.(ir.Instruction)] = struct{}{}
				}
				seen2[ins] = struct{}{}
				failed := false
				for v := range seen {
					if !validateReferrers(v, seen2) {
						failed = true
						break
					}
				}
				if !failed {
					report.Report(pass, ins, "this result of append is never used, except maybe in other appends")
				}
			}
		}
	}
	return nil, nil
}

func CheckConcurrentTesting(pass *analysis.Pass) (interface{}, error) {
	for _, fn := range pass.ResultOf[buildir.Analyzer].(*buildir.IR).SrcFuncs {
		for _, block := range fn.Blocks {
			for _, ins := range block.Instrs {
				gostmt, ok := ins.(*ir.Go)
				if !ok {
					continue
				}
				var fn *ir.Function
				switch val := gostmt.Call.Value.(type) {
				case *ir.Function:
					fn = val
				case *ir.MakeClosure:
					fn = val.Fn.(*ir.Function)
				default:
					continue
				}
				if fn.Blocks == nil {
					continue
				}
				for _, block := range fn.Blocks {
					for _, ins := range block.Instrs {
						call, ok := ins.(*ir.Call)
						if !ok {
							continue
						}
						if call.Call.IsInvoke() {
							continue
						}
						callee := call.Call.StaticCallee()
						if callee == nil {
							continue
						}
						recv := callee.Signature.Recv()
						if recv == nil {
							continue
						}
						if !typeutil.IsType(recv.Type(), "*testing.common") {
							continue
						}
						fn, ok := call.Call.StaticCallee().Object().(*types.Func)
						if !ok {
							continue
						}
						name := fn.Name()
						switch name {
						case "FailNow", "Fatal", "Fatalf", "SkipNow", "Skip", "Skipf":
						default:
							continue
						}
						// TODO(dh): don't report multiple diagnostics
						// for multiple calls to T.Fatal, but do
						// collect all of them as related information
						report.Report(pass, gostmt, fmt.Sprintf("the goroutine calls T.%s, which must be called in the same goroutine as the test", name),
							report.Related(call, fmt.Sprintf("call to T.%s", name)))
					}
				}
			}
		}
	}
	return nil, nil
}

func eachCall(fn *ir.Function, cb func(caller *ir.Function, site ir.CallInstruction, callee *ir.Function)) {
	for _, b := range fn.Blocks {
		for _, instr := range b.Instrs {
			if site, ok := instr.(ir.CallInstruction); ok {
				if g := site.Common().StaticCallee(); g != nil {
					cb(fn, site, g)
				}
			}
		}
	}
}

func CheckCyclicFinalizer(pass *analysis.Pass) (interface{}, error) {
	cb := func(caller *ir.Function, site ir.CallInstruction, callee *ir.Function) {
		if callee.RelString(nil) != "runtime.SetFinalizer" {
			return
		}
		arg0 := site.Common().Args[knowledge.Arg("runtime.SetFinalizer.obj")]
		if iface, ok := arg0.(*ir.MakeInterface); ok {
			arg0 = iface.X
		}
		load, ok := arg0.(*ir.Load)
		if !ok {
			return
		}
		v, ok := load.X.(*ir.Alloc)
		if !ok {
			return
		}
		arg1 := site.Common().Args[knowledge.Arg("runtime.SetFinalizer.finalizer")]
		if iface, ok := arg1.(*ir.MakeInterface); ok {
			arg1 = iface.X
		}
		mc, ok := arg1.(*ir.MakeClosure)
		if !ok {
			return
		}
		for _, b := range mc.Bindings {
			if b == v {
				pos := report.DisplayPosition(pass.Fset, mc.Fn.Pos())
				report.Report(pass, site, fmt.Sprintf("the finalizer closes over the object, preventing the finalizer from ever running (at %s)", pos))
			}
		}
	}
	for _, fn := range pass.ResultOf[buildir.Analyzer].(*buildir.IR).SrcFuncs {
		eachCall(fn, cb)
	}
	return nil, nil
}

/*
func CheckSliceOutOfBounds(pass *analysis.Pass) (interface{}, error) {
	for _, fn := range pass.ResultOf[buildir.Analyzer].(*buildir.IR).SrcFuncs {
		for _, block := range fn.Blocks {
			for _, ins := range block.Instrs {
				ia, ok := ins.(*ir.IndexAddr)
				if !ok {
					continue
				}
				if _, ok := ia.X.Type().Underlying().(*types.Slice); !ok {
					continue
				}
				sr, ok1 := c.funcDescs.Get(fn).Ranges[ia.X].(vrp.SliceInterval)
				idxr, ok2 := c.funcDescs.Get(fn).Ranges[ia.Index].(vrp.IntInterval)
				if !ok1 || !ok2 || !sr.IsKnown() || !idxr.IsKnown() || sr.Length.Empty() || idxr.Empty() {
					continue
				}
				if idxr.Lower.Cmp(sr.Length.Upper) >= 0 {
					report.Nodef(pass, ia, "index out of bounds")
				}
			}
		}
	}
	return nil, nil
}
*/

func CheckDeferLock(pass *analysis.Pass) (interface{}, error) {
	for _, fn := range pass.ResultOf[buildir.Analyzer].(*buildir.IR).SrcFuncs {
		for _, block := range fn.Blocks {
			instrs := irutil.FilterDebug(block.Instrs)
			if len(instrs) < 2 {
				continue
			}
			for i, ins := range instrs[:len(instrs)-1] {
				call, ok := ins.(*ir.Call)
				if !ok {
					continue
				}
				if !irutil.IsCallToAny(call.Common(), "(*sync.Mutex).Lock", "(*sync.RWMutex).RLock") {
					continue
				}
				nins, ok := instrs[i+1].(*ir.Defer)
				if !ok {
					continue
				}
				if !irutil.IsCallToAny(&nins.Call, "(*sync.Mutex).Lock", "(*sync.RWMutex).RLock") {
					continue
				}
				if call.Common().Args[0] != nins.Call.Args[0] {
					continue
				}
				name := shortCallName(call.Common())
				alt := ""
				switch name {
				case "Lock":
					alt = "Unlock"
				case "RLock":
					alt = "RUnlock"
				}
				report.Report(pass, nins, fmt.Sprintf("deferring %s right after having locked already; did you mean to defer %s?", name, alt))
			}
		}
	}
	return nil, nil
}

func CheckNaNComparison(pass *analysis.Pass) (interface{}, error) {
	isNaN := func(v ir.Value) bool {
		call, ok := v.(*ir.Call)
		if !ok {
			return false
		}
		return irutil.IsCallTo(call.Common(), "math.NaN")
	}
	for _, fn := range pass.ResultOf[buildir.Analyzer].(*buildir.IR).SrcFuncs {
		for _, block := range fn.Blocks {
			for _, ins := range block.Instrs {
				ins, ok := ins.(*ir.BinOp)
				if !ok {
					continue
				}
				if isNaN(irutil.Flatten(ins.X)) || isNaN(irutil.Flatten(ins.Y)) {
					report.Report(pass, ins, "no value is equal to NaN, not even NaN itself")
				}
			}
		}
	}
	return nil, nil
}

func CheckInfiniteRecursion(pass *analysis.Pass) (interface{}, error) {
	for _, fn := range pass.ResultOf[buildir.Analyzer].(*buildir.IR).SrcFuncs {
		eachCall(fn, func(caller *ir.Function, site ir.CallInstruction, callee *ir.Function) {
			if callee != fn {
				return
			}
			if _, ok := site.(*ir.Go); ok {
				// Recursively spawning goroutines doesn't consume
				// stack space infinitely, so don't flag it.
				return
			}

			block := site.Block()
			canReturn := false
			for _, b := range fn.Blocks {
				if block.Dominates(b) {
					continue
				}
				if len(b.Instrs) == 0 {
					continue
				}
				if _, ok := b.Control().(*ir.Return); ok {
					canReturn = true
					break
				}
			}
			if canReturn {
				return
			}
			report.Report(pass, site, "infinite recursive call")
		})
	}
	return nil, nil
}

func objectName(obj types.Object) string {
	if obj == nil {
		return "<nil>"
	}
	var name string
	if obj.Pkg() != nil && obj.Pkg().Scope().Lookup(obj.Name()) == obj {
		s := obj.Pkg().Path()
		if s != "" {
			name += s + "."
		}
	}
	name += obj.Name()
	return name
}

func isName(pass *analysis.Pass, expr ast.Expr, name string) bool {
	var obj types.Object
	switch expr := expr.(type) {
	case *ast.Ident:
		obj = pass.TypesInfo.ObjectOf(expr)
	case *ast.SelectorExpr:
		obj = pass.TypesInfo.ObjectOf(expr.Sel)
	}
	return objectName(obj) == name
}

func CheckLeakyTimeTick(pass *analysis.Pass) (interface{}, error) {
	for _, fn := range pass.ResultOf[buildir.Analyzer].(*buildir.IR).SrcFuncs {
		if code.IsMainLike(pass) || code.IsInTest(pass, fn) {
			continue
		}
		for _, block := range fn.Blocks {
			for _, ins := range block.Instrs {
				call, ok := ins.(*ir.Call)
				if !ok || !irutil.IsCallTo(call.Common(), "time.Tick") {
					continue
				}
				if !irutil.Terminates(call.Parent()) {
					continue
				}
				report.Report(pass, call, "using time.Tick leaks the underlying ticker, consider using it only in endless functions, tests and the main package, and use time.NewTicker here")
			}
		}
	}
	return nil, nil
}

var checkDoubleNegationQ = pattern.MustParse(`(UnaryExpr "!" single@(UnaryExpr "!" x))`)

func CheckDoubleNegation(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		if m, ok := code.Match(pass, checkDoubleNegationQ, node); ok {
			report.Report(pass, node, "negating a boolean twice has no effect; is this a typo?", report.Fixes(
				edit.Fix("turn into single negation", edit.ReplaceWithNode(pass.Fset, node, m.State["single"].(ast.Node))),
				edit.Fix("remove double negation", edit.ReplaceWithNode(pass.Fset, node, m.State["x"].(ast.Node)))))
		}
	}
	code.Preorder(pass, fn, (*ast.UnaryExpr)(nil))
	return nil, nil
}

func CheckRepeatedIfElse(pass *analysis.Pass) (interface{}, error) {
	seen := map[ast.Node]bool{}

	var collectConds func(ifstmt *ast.IfStmt, conds []ast.Expr) ([]ast.Expr, bool)
	collectConds = func(ifstmt *ast.IfStmt, conds []ast.Expr) ([]ast.Expr, bool) {
		seen[ifstmt] = true
		// Bail if any if-statement has an Init statement or side effects in its condition
		if ifstmt.Init != nil {
			return nil, false
		}
		if code.MayHaveSideEffects(pass, ifstmt.Cond, nil) {
			return nil, false
		}

		conds = append(conds, ifstmt.Cond)
		if elsestmt, ok := ifstmt.Else.(*ast.IfStmt); ok {
			return collectConds(elsestmt, conds)
		}
		return conds, true
	}
	fn := func(node ast.Node) {
		ifstmt := node.(*ast.IfStmt)
		if seen[ifstmt] {
			// this if-statement is part of an if/else-if chain that we've already processed
			return
		}
		if ifstmt.Else == nil {
			// there can be at most one condition
			return
		}
		conds, ok := collectConds(ifstmt, nil)
		if !ok {
			return
		}
		if len(conds) < 2 {
			return
		}
		counts := map[string]int{}
		for _, cond := range conds {
			s := report.Render(pass, cond)
			counts[s]++
			if counts[s] == 2 {
				report.Report(pass, cond, "this condition occurs multiple times in this if/else if chain")
			}
		}
	}
	code.Preorder(pass, fn, (*ast.IfStmt)(nil))
	return nil, nil
}

func CheckSillyBitwiseOps(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		binop := node.(*ast.BinaryExpr)
		b, ok := pass.TypesInfo.TypeOf(binop).Underlying().(*types.Basic)
		if !ok {
			return
		}
		if (b.Info() & types.IsInteger) == 0 {
			return
		}
		switch binop.Op {
		case token.AND, token.OR, token.XOR:
		default:
			// we do not flag shifts because too often, x<<0 is part
			// of a pattern, x<<0, x<<8, x<<16, ...
			return
		}
		switch y := binop.Y.(type) {
		case *ast.Ident:
			obj, ok := pass.TypesInfo.ObjectOf(y).(*types.Const)
			if !ok {
				return
			}
			if obj.Pkg() != pass.Pkg {
				// identifier was dot-imported
				return
			}
			if v, _ := constant.Int64Val(obj.Val()); v != 0 {
				return
			}
			path, _ := astutil.PathEnclosingInterval(code.File(pass, obj), obj.Pos(), obj.Pos())
			if len(path) < 2 {
				return
			}
			spec, ok := path[1].(*ast.ValueSpec)
			if !ok {
				return
			}
			if len(spec.Names) != 1 || len(spec.Values) != 1 {
				// TODO(dh): we could support this
				return
			}
			ident, ok := spec.Values[0].(*ast.Ident)
			if !ok {
				return
			}
			if !isIota(pass.TypesInfo.ObjectOf(ident)) {
				return
			}
			switch binop.Op {
			case token.AND:
				report.Report(pass, node,
					fmt.Sprintf("%s always equals 0; %s is defined as iota and has value 0, maybe %s is meant to be 1 << iota?", report.Render(pass, binop), report.Render(pass, binop.Y), report.Render(pass, binop.Y)))
			case token.OR, token.XOR:
				report.Report(pass, node,
					fmt.Sprintf("%s always equals %s; %s is defined as iota and has value 0, maybe %s is meant to be 1 << iota?", report.Render(pass, binop), report.Render(pass, binop.X), report.Render(pass, binop.Y), report.Render(pass, binop.Y)))
			}
		case *ast.BasicLit:
			if !astutil.IsIntLiteral(binop.Y, "0") {
				return
			}
			switch binop.Op {
			case token.AND:
				report.Report(pass, node, fmt.Sprintf("%s always equals 0", report.Render(pass, binop)))
			case token.OR, token.XOR:
				report.Report(pass, node, fmt.Sprintf("%s always equals %s", report.Render(pass, binop), report.Render(pass, binop.X)))
			}
		default:
			return
		}
	}
	code.Preorder(pass, fn, (*ast.BinaryExpr)(nil))
	return nil, nil
}

func isIota(obj types.Object) bool {
	if obj.Name() != "iota" {
		return false
	}
	c, ok := obj.(*types.Const)
	if !ok {
		return false
	}
	return c.Pkg() == nil
}

func CheckNonOctalFileMode(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		call := node.(*ast.CallExpr)
		for _, arg := range call.Args {
			lit, ok := arg.(*ast.BasicLit)
			if !ok {
				continue
			}
			if !typeutil.IsType(pass.TypesInfo.TypeOf(lit), "os.FileMode") &&
				!typeutil.IsType(pass.TypesInfo.TypeOf(lit), "io/fs.FileMode") {
				continue
			}
			if len(lit.Value) == 3 &&
				lit.Value[0] != '0' &&
				lit.Value[0] >= '0' && lit.Value[0] <= '7' &&
				lit.Value[1] >= '0' && lit.Value[1] <= '7' &&
				lit.Value[2] >= '0' && lit.Value[2] <= '7' {

				v, err := strconv.ParseInt(lit.Value, 10, 64)
				if err != nil {
					continue
				}
				report.Report(pass, arg, fmt.Sprintf("file mode '%s' evaluates to %#o; did you mean '0%s'?", lit.Value, v, lit.Value),
					report.Fixes(edit.Fix("fix octal literal", edit.ReplaceWithString(pass.Fset, arg, "0"+lit.Value))))
			}
		}
	}
	code.Preorder(pass, fn, (*ast.CallExpr)(nil))
	return nil, nil
}

func CheckPureFunctions(pass *analysis.Pass) (interface{}, error) {
	pure := pass.ResultOf[facts.Purity].(facts.PurityResult)

fnLoop:
	for _, fn := range pass.ResultOf[buildir.Analyzer].(*buildir.IR).SrcFuncs {
		if code.IsInTest(pass, fn) {
			params := fn.Signature.Params()
			for i := 0; i < params.Len(); i++ {
				param := params.At(i)
				if typeutil.IsType(param.Type(), "*testing.B") {
					// Ignore discarded pure functions in code related
					// to benchmarks. Instead of matching BenchmarkFoo
					// functions, we match any function accepting a
					// *testing.B. Benchmarks sometimes call generic
					// functions for doing the actual work, and
					// checking for the parameter is a lot easier and
					// faster than analyzing call trees.
					continue fnLoop
				}
			}
		}

		for _, b := range fn.Blocks {
			for _, ins := range b.Instrs {
				ins, ok := ins.(*ir.Call)
				if !ok {
					continue
				}
				refs := ins.Referrers()
				if refs == nil || len(irutil.FilterDebug(*refs)) > 0 {
					continue
				}

				callee := ins.Common().StaticCallee()
				if callee == nil {
					continue
				}
				if callee.Object() == nil {
					// TODO(dh): support anonymous functions
					continue
				}
				if _, ok := pure[callee.Object().(*types.Func)]; ok {
					if pass.Pkg.Path() == "fmt_test" && callee.Object().(*types.Func).FullName() == "fmt.Sprintf" {
						// special case for benchmarks in the fmt package
						continue
					}
					report.Report(pass, ins, fmt.Sprintf("%s is a pure function but its return value is ignored", callee.Name()))
				}
			}
		}
	}
	return nil, nil
}

func CheckDeprecated(pass *analysis.Pass) (interface{}, error) {
	deprs := pass.ResultOf[facts.Deprecated].(facts.DeprecatedResult)

	// Selectors can appear outside of function literals, e.g. when
	// declaring package level variables.

	var tfn types.Object
	stack := 0
	fn := func(node ast.Node, push bool) bool {
		if !push {
			stack--
			return false
		}
		stack++
		if stack == 1 {
			tfn = nil
		}
		if fn, ok := node.(*ast.FuncDecl); ok {
			tfn = pass.TypesInfo.ObjectOf(fn.Name)
		}
		sel, ok := node.(*ast.SelectorExpr)
		if !ok {
			return true
		}

		obj := pass.TypesInfo.ObjectOf(sel.Sel)
		if obj.Pkg() == nil {
			return true
		}
		if pass.Pkg == obj.Pkg() || obj.Pkg().Path()+"_test" == pass.Pkg.Path() {
			// Don't flag stuff in our own package
			return true
		}
		if depr, ok := deprs.Objects[obj]; ok {
			// Note: gopls doesn't correctly run analyzers on
			// dependencies, so we'll never be able to find deprecated
			// objects in imported code. We've experimented with
			// lifting the stdlib handling out of the general check,
			// to at least work for deprecated objects in the stdlib,
			// but we gave up on that, because we wouldn't have access
			// to the deprecation message.
			std, ok := knowledge.StdlibDeprecations[code.SelectorName(pass, sel)]
			if ok {
				switch std.AlternativeAvailableSince {
				case knowledge.DeprecatedNeverUse:
					// This should never be used, regardless of the
					// targeted Go version. Examples include insecure
					// cryptography or inherently broken APIs.
					//
					// We always want to flag these.
				case knowledge.DeprecatedUseNoLonger:
					// This should no longer be used. Using it with
					// older Go versions might still make sense.
					if !code.IsGoVersion(pass, std.DeprecatedSince) {
						return true
					}
				default:
					if std.AlternativeAvailableSince < 0 {
						panic(fmt.Sprintf("unhandled case %d", std.AlternativeAvailableSince))
					}
					// Look for the first available alternative, not the first
					// version something was deprecated in. If a function was
					// deprecated in Go 1.6, an alternative has been available
					// already in 1.0, and we're targeting 1.2, it still
					// makes sense to use the alternative from 1.0, to be
					// future-proof.
					if !code.IsGoVersion(pass, std.AlternativeAvailableSince) {
						return true
					}
				}
			}

			if tfn != nil {
				if _, ok := deprs.Objects[tfn]; ok {
					// functions that are deprecated may use deprecated
					// symbols
					return true
				}
			}

			if ok {
				if std.AlternativeAvailableSince == knowledge.DeprecatedNeverUse {
					report.Report(pass, sel, fmt.Sprintf("%s has been deprecated since Go 1.%d because it shouldn't be used: %s", report.Render(pass, sel), std.DeprecatedSince, depr.Msg))
				} else if std.AlternativeAvailableSince == std.DeprecatedSince || std.AlternativeAvailableSince == knowledge.DeprecatedUseNoLonger {
					report.Report(pass, sel, fmt.Sprintf("%s has been deprecated since Go 1.%d: %s", report.Render(pass, sel), std.DeprecatedSince, depr.Msg))
				} else {
					report.Report(pass, sel, fmt.Sprintf("%s has been deprecated since Go 1.%d and an alternative has been available since Go 1.%d: %s", report.Render(pass, sel), std.DeprecatedSince, std.AlternativeAvailableSince, depr.Msg))
				}
			} else {
				report.Report(pass, sel, fmt.Sprintf("%s is deprecated: %s", report.Render(pass, sel), depr.Msg))
			}
			return true
		}
		return true
	}

	fn2 := func(node ast.Node) {
		spec := node.(*ast.ImportSpec)
		var imp *types.Package
		if spec.Name != nil {
			imp = pass.TypesInfo.ObjectOf(spec.Name).(*types.PkgName).Imported()
		} else {
			imp = pass.TypesInfo.Implicits[spec].(*types.PkgName).Imported()
		}

		p := spec.Path.Value
		path := p[1 : len(p)-1]
		if depr, ok := deprs.Packages[imp]; ok {
			if path == "github.com/golang/protobuf/proto" {
				gen, ok := code.Generator(pass, spec.Path.Pos())
				if ok && gen == facts.ProtocGenGo {
					return
				}
			}
			report.Report(pass, spec, fmt.Sprintf("package %s is deprecated: %s", path, depr.Msg))
		}
	}
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Nodes(nil, fn)
	code.Preorder(pass, fn2, (*ast.ImportSpec)(nil))
	return nil, nil
}

func callChecker(rules map[string]CallCheck) func(pass *analysis.Pass) (interface{}, error) {
	return func(pass *analysis.Pass) (interface{}, error) {
		return checkCalls(pass, rules)
	}
}

func checkCalls(pass *analysis.Pass, rules map[string]CallCheck) (interface{}, error) {
	cb := func(caller *ir.Function, site ir.CallInstruction, callee *ir.Function) {
		obj, ok := callee.Object().(*types.Func)
		if !ok {
			return
		}

		r, ok := rules[typeutil.FuncName(obj)]
		if !ok {
			return
		}
		var args []*Argument
		irargs := site.Common().Args
		if callee.Signature.Recv() != nil {
			irargs = irargs[1:]
		}
		for _, arg := range irargs {
			if iarg, ok := arg.(*ir.MakeInterface); ok {
				arg = iarg.X
			}
			args = append(args, &Argument{Value: Value{arg}})
		}
		call := &Call{
			Pass:   pass,
			Instr:  site,
			Args:   args,
			Parent: site.Parent(),
		}
		r(call)

		var astcall *ast.CallExpr
		switch source := site.Source().(type) {
		case *ast.CallExpr:
			astcall = source
		case *ast.DeferStmt:
			astcall = source.Call
		case *ast.GoStmt:
			astcall = source.Call
		case nil:
			// TODO(dh): I am not sure this can actually happen. If it
			// can't, we should remove this case, and also stop
			// checking for astcall == nil in the code that follows.
		default:
			panic(fmt.Sprintf("unhandled case %T", source))
		}

		for idx, arg := range call.Args {
			for _, e := range arg.invalids {
				if astcall != nil {
					if idx < len(astcall.Args) {
						report.Report(pass, astcall.Args[idx], e)
					} else {
						// this is an instance of fn1(fn2()) where fn2
						// returns multiple values. Report the error
						// at the next-best position that we have, the
						// first argument. An example of a check that
						// triggers this is checkEncodingBinaryRules.
						report.Report(pass, astcall.Args[0], e)
					}
				} else {
					report.Report(pass, site, e)
				}
			}
		}
		for _, e := range call.invalids {
			report.Report(pass, call.Instr, e)
		}
	}
	for _, fn := range pass.ResultOf[buildir.Analyzer].(*buildir.IR).SrcFuncs {
		eachCall(fn, cb)
	}
	return nil, nil
}

func shortCallName(call *ir.CallCommon) string {
	if call.IsInvoke() {
		return ""
	}
	switch v := call.Value.(type) {
	case *ir.Function:
		fn, ok := v.Object().(*types.Func)
		if !ok {
			return ""
		}
		return fn.Name()
	case *ir.Builtin:
		return v.Name()
	}
	return ""
}

func CheckWriterBufferModified(pass *analysis.Pass) (interface{}, error) {
	// TODO(dh): this might be a good candidate for taint analysis.
	// Taint the argument as MUST_NOT_MODIFY, then propagate that
	// through functions like bytes.Split

	for _, fn := range pass.ResultOf[buildir.Analyzer].(*buildir.IR).SrcFuncs {
		sig := fn.Signature
		if fn.Name() != "Write" || sig.Recv() == nil || sig.Params().Len() != 1 || sig.Results().Len() != 2 {
			continue
		}
		tArg, ok := sig.Params().At(0).Type().(*types.Slice)
		if !ok {
			continue
		}
		if basic, ok := tArg.Elem().(*types.Basic); !ok || basic.Kind() != types.Byte {
			continue
		}
		if basic, ok := sig.Results().At(0).Type().(*types.Basic); !ok || basic.Kind() != types.Int {
			continue
		}
		if named, ok := sig.Results().At(1).Type().(*types.Named); !ok || !typeutil.IsType(named, "error") {
			continue
		}

		for _, block := range fn.Blocks {
			for _, ins := range block.Instrs {
				switch ins := ins.(type) {
				case *ir.Store:
					addr, ok := ins.Addr.(*ir.IndexAddr)
					if !ok {
						continue
					}
					if addr.X != fn.Params[1] {
						continue
					}
					report.Report(pass, ins, "io.Writer.Write must not modify the provided buffer, not even temporarily")
				case *ir.Call:
					if !irutil.IsCallTo(ins.Common(), "append") {
						continue
					}
					if ins.Common().Args[0] != fn.Params[1] {
						continue
					}
					report.Report(pass, ins, "io.Writer.Write must not modify the provided buffer, not even temporarily")
				}
			}
		}
	}
	return nil, nil
}

func loopedRegexp(name string) CallCheck {
	return func(call *Call) {
		if extractConst(call.Args[0].Value.Value) == nil {
			return
		}
		if !isInLoop(call.Instr.Block()) {
			return
		}
		call.Invalid(fmt.Sprintf("calling %s in a loop has poor performance, consider using regexp.Compile", name))
	}
}

func CheckEmptyBranch(pass *analysis.Pass) (interface{}, error) {
	for _, fn := range pass.ResultOf[buildir.Analyzer].(*buildir.IR).SrcFuncs {
		if fn.Source() == nil {
			continue
		}
		if irutil.IsExample(fn) {
			continue
		}
		cb := func(node ast.Node) bool {
			ifstmt, ok := node.(*ast.IfStmt)
			if !ok {
				return true
			}
			if ifstmt.Else != nil {
				b, ok := ifstmt.Else.(*ast.BlockStmt)
				if !ok || len(b.List) != 0 {
					return true
				}
				report.Report(pass, ifstmt.Else, "empty branch", report.FilterGenerated(), report.ShortRange())
			}
			if len(ifstmt.Body.List) != 0 {
				return true
			}
			report.Report(pass, ifstmt, "empty branch", report.FilterGenerated(), report.ShortRange())
			return true
		}
		if source := fn.Source(); source != nil {
			ast.Inspect(source, cb)
		}
	}
	return nil, nil
}

func CheckMapBytesKey(pass *analysis.Pass) (interface{}, error) {
	for _, fn := range pass.ResultOf[buildir.Analyzer].(*buildir.IR).SrcFuncs {
		for _, b := range fn.Blocks {
		insLoop:
			for _, ins := range b.Instrs {
				// find []byte -> string conversions
				conv, ok := ins.(*ir.Convert)
				if !ok || conv.Type() != types.Universe.Lookup("string").Type() {
					continue
				}
				if s, ok := conv.X.Type().(*types.Slice); !ok || s.Elem() != types.Universe.Lookup("byte").Type() {
					continue
				}
				refs := conv.Referrers()
				// need at least two (DebugRef) references: the
				// conversion and the *ast.Ident
				if refs == nil || len(*refs) < 2 {
					continue
				}
				ident := false
				// skip first reference, that's the conversion itself
				for _, ref := range (*refs)[1:] {
					switch ref := ref.(type) {
					case *ir.DebugRef:
						if _, ok := ref.Expr.(*ast.Ident); !ok {
							// the string seems to be used somewhere
							// unexpected; the default branch should
							// catch this already, but be safe
							continue insLoop
						} else {
							ident = true
						}
					case *ir.MapLookup:
					default:
						// the string is used somewhere else than a
						// map lookup
						continue insLoop
					}
				}

				// the result of the conversion wasn't assigned to an
				// identifier
				if !ident {
					continue
				}
				report.Report(pass, conv, "m[string(key)] would be more efficient than k := string(key); m[k]")
			}
		}
	}
	return nil, nil
}

func CheckRangeStringRunes(pass *analysis.Pass) (interface{}, error) {
	return sharedcheck.CheckRangeStringRunes(pass)
}

func CheckSelfAssignment(pass *analysis.Pass) (interface{}, error) {
	pure := pass.ResultOf[facts.Purity].(facts.PurityResult)

	fn := func(node ast.Node) {
		assign := node.(*ast.AssignStmt)
		if assign.Tok != token.ASSIGN || len(assign.Lhs) != len(assign.Rhs) {
			return
		}
		for i, lhs := range assign.Lhs {
			rhs := assign.Rhs[i]
			if reflect.TypeOf(lhs) != reflect.TypeOf(rhs) {
				continue
			}
			if code.MayHaveSideEffects(pass, lhs, pure) || code.MayHaveSideEffects(pass, rhs, pure) {
				continue
			}

			rlh := report.Render(pass, lhs)
			rrh := report.Render(pass, rhs)
			if rlh == rrh {
				report.Report(pass, assign, fmt.Sprintf("self-assignment of %s to %s", rrh, rlh), report.FilterGenerated())
			}
		}
	}
	code.Preorder(pass, fn, (*ast.AssignStmt)(nil))
	return nil, nil
}

func buildTagsIdentical(s1, s2 []string) bool {
	if len(s1) != len(s2) {
		return false
	}
	s1s := make([]string, len(s1))
	copy(s1s, s1)
	sort.Strings(s1s)
	s2s := make([]string, len(s2))
	copy(s2s, s2)
	sort.Strings(s2s)
	for i, s := range s1s {
		if s != s2s[i] {
			return false
		}
	}
	return true
}

func CheckDuplicateBuildConstraints(pass *analysis.Pass) (interface{}, error) {
	for _, f := range pass.Files {
		constraints := buildTags(f)
		for i, constraint1 := range constraints {
			for j, constraint2 := range constraints {
				if i >= j {
					continue
				}
				if buildTagsIdentical(constraint1, constraint2) {
					msg := fmt.Sprintf("identical build constraints %q and %q",
						strings.Join(constraint1, " "),
						strings.Join(constraint2, " "))
					report.Report(pass, f, msg, report.FilterGenerated(), report.ShortRange())
				}
			}
		}
	}
	return nil, nil
}

func CheckSillyRegexp(pass *analysis.Pass) (interface{}, error) {
	// We could use the rule checking engine for this, but the
	// arguments aren't really invalid.
	for _, fn := range pass.ResultOf[buildir.Analyzer].(*buildir.IR).SrcFuncs {
		for _, b := range fn.Blocks {
			for _, ins := range b.Instrs {
				call, ok := ins.(*ir.Call)
				if !ok {
					continue
				}
				if !irutil.IsCallToAny(call.Common(), "regexp.MustCompile", "regexp.Compile", "regexp.Match", "regexp.MatchReader", "regexp.MatchString") {
					continue
				}
				c, ok := call.Common().Args[0].(*ir.Const)
				if !ok {
					continue
				}
				s := constant.StringVal(c.Value)
				re, err := syntax.Parse(s, 0)
				if err != nil {
					continue
				}
				if re.Op != syntax.OpLiteral && re.Op != syntax.OpEmptyMatch {
					continue
				}
				report.Report(pass, call, "regular expression does not contain any meta characters")
			}
		}
	}
	return nil, nil
}

func CheckMissingEnumTypesInDeclaration(pass *analysis.Pass) (interface{}, error) {
	convertibleTo := func(V, T types.Type) bool {
		if types.ConvertibleTo(V, T) {
			return true
		}
		// Go <1.16 returns false for untyped string to string conversion
		if V, ok := V.(*types.Basic); ok && V.Kind() == types.UntypedString {
			if T, ok := T.Underlying().(*types.Basic); ok && T.Kind() == types.String {
				return true
			}
		}
		return false
	}
	fn := func(node ast.Node) {
		decl := node.(*ast.GenDecl)
		if !decl.Lparen.IsValid() {
			return
		}
		if decl.Tok != token.CONST {
			return
		}

		groups := astutil.GroupSpecs(pass.Fset, decl.Specs)
	groupLoop:
		for _, group := range groups {
			if len(group) < 2 {
				continue
			}
			if group[0].(*ast.ValueSpec).Type == nil {
				// first constant doesn't have a type
				continue groupLoop
			}

			firstType := pass.TypesInfo.TypeOf(group[0].(*ast.ValueSpec).Values[0])
			for i, spec := range group {
				spec := spec.(*ast.ValueSpec)
				if i > 0 && spec.Type != nil {
					continue groupLoop
				}
				if len(spec.Names) != 1 || len(spec.Values) != 1 {
					continue groupLoop
				}

				if !convertibleTo(pass.TypesInfo.TypeOf(spec.Values[0]), firstType) {
					continue groupLoop
				}

				switch v := spec.Values[0].(type) {
				case *ast.BasicLit:
				case *ast.UnaryExpr:
					if _, ok := v.X.(*ast.BasicLit); !ok {
						continue groupLoop
					}
				default:
					// if it's not a literal it might be typed, such as
					// time.Microsecond = 1000 * Nanosecond
					continue groupLoop
				}
			}
			var edits []analysis.TextEdit
			typ := group[0].(*ast.ValueSpec).Type
			for _, spec := range group[1:] {
				nspec := *spec.(*ast.ValueSpec)
				nspec.Type = typ
				// The position of `spec` node excludes comments (if any).
				// However, on generating the source back from the node, the comments are included. Setting `Comment` to nil ensures deduplication of comments.
				nspec.Comment = nil
				edits = append(edits, edit.ReplaceWithNode(pass.Fset, spec, &nspec))
			}
			report.Report(pass, group[0], "only the first constant in this group has an explicit type", report.Fixes(edit.Fix("add type to all constants in group", edits...)))
		}
	}
	code.Preorder(pass, fn, (*ast.GenDecl)(nil))
	return nil, nil
}

func CheckTimerResetReturnValue(pass *analysis.Pass) (interface{}, error) {
	for _, fn := range pass.ResultOf[buildir.Analyzer].(*buildir.IR).SrcFuncs {
		for _, block := range fn.Blocks {
			for _, ins := range block.Instrs {
				call, ok := ins.(*ir.Call)
				if !ok {
					continue
				}
				if !irutil.IsCallTo(call.Common(), "(*time.Timer).Reset") {
					continue
				}
				refs := call.Referrers()
				if refs == nil {
					continue
				}
				for _, ref := range irutil.FilterDebug(*refs) {
					ifstmt, ok := ref.(*ir.If)
					if !ok {
						continue
					}

					found := false
					for _, succ := range ifstmt.Block().Succs {
						if len(succ.Preds) != 1 {
							// Merge point, not a branch in the
							// syntactical sense.

							// FIXME(dh): this is broken for if
							// statements a la "if x || y"
							continue
						}
						irutil.Walk(succ, func(b *ir.BasicBlock) bool {
							if !succ.Dominates(b) {
								// We've reached the end of the branch
								return false
							}
							for _, ins := range b.Instrs {
								// TODO(dh): we should check that
								// we're receiving from the channel of
								// a time.Timer to further reduce
								// false positives. Not a key
								// priority, considering the rarity of
								// Reset and the tiny likeliness of a
								// false positive
								if ins, ok := ins.(*ir.Recv); ok && typeutil.IsType(ins.Chan.Type(), "<-chan time.Time") {
									found = true
									return false
								}
							}
							return true
						})
					}

					if found {
						report.Report(pass, call, "it is not possible to use Reset's return value correctly, as there is a race condition between draining the channel and the new timer expiring")
					}
				}
			}
		}
	}
	return nil, nil
}

var (
	checkToLowerToUpperComparisonQ = pattern.MustParse(`
	(BinaryExpr
		(CallExpr fun@(Function (Or "strings.ToLower" "strings.ToUpper")) [a])
 		tok@(Or "==" "!=")
 		(CallExpr fun [b]))`)
	checkToLowerToUpperComparisonR = pattern.MustParse(`(CallExpr (SelectorExpr (Ident "strings") (Ident "EqualFold")) [a b])`)
)

func CheckToLowerToUpperComparison(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		m, ok := code.Match(pass, checkToLowerToUpperComparisonQ, node)
		if !ok {
			return
		}
		rn := pattern.NodeToAST(checkToLowerToUpperComparisonR.Root, m.State).(ast.Expr)
		if m.State["tok"].(token.Token) == token.NEQ {
			rn = &ast.UnaryExpr{
				Op: token.NOT,
				X:  rn,
			}
		}

		report.Report(pass, node, "should use strings.EqualFold instead", report.Fixes(edit.Fix("replace with strings.EqualFold", edit.ReplaceWithNode(pass.Fset, node, rn))))
	}

	code.Preorder(pass, fn, (*ast.BinaryExpr)(nil))
	return nil, nil
}

func CheckUnreachableTypeCases(pass *analysis.Pass) (interface{}, error) {
	// Check if T subsumes V in a type switch. T subsumes V if T is an interface and T's method set is a subset of V's method set.
	subsumes := func(T, V types.Type) bool {
		tIface, ok := T.Underlying().(*types.Interface)
		if !ok {
			return false
		}

		return types.Implements(V, tIface)
	}

	subsumesAny := func(Ts, Vs []types.Type) (types.Type, types.Type, bool) {
		for _, T := range Ts {
			for _, V := range Vs {
				if subsumes(T, V) {
					return T, V, true
				}
			}
		}

		return nil, nil, false
	}

	fn := func(node ast.Node) {
		tsStmt := node.(*ast.TypeSwitchStmt)

		type ccAndTypes struct {
			cc    *ast.CaseClause
			types []types.Type
		}

		// All asserted types in the order of case clauses.
		ccs := make([]ccAndTypes, 0, len(tsStmt.Body.List))
		for _, stmt := range tsStmt.Body.List {
			cc, _ := stmt.(*ast.CaseClause)

			// Exclude the 'default' case.
			if len(cc.List) == 0 {
				continue
			}

			Ts := make([]types.Type, 0, len(cc.List))
			for _, expr := range cc.List {
				// Exclude the 'nil' value from any 'case' statement (it is always reachable).
				if typ := pass.TypesInfo.TypeOf(expr); typ != types.Typ[types.UntypedNil] {
					Ts = append(Ts, typ)
				}
			}

			ccs = append(ccs, ccAndTypes{cc: cc, types: Ts})
		}

		if len(ccs) <= 1 {
			// Zero or one case clauses, nothing to check.
			return
		}

		// Check if case clauses following cc have types that are subsumed by cc.
		for i, cc := range ccs[:len(ccs)-1] {
			for _, next := range ccs[i+1:] {
				if T, V, yes := subsumesAny(cc.types, next.types); yes {
					report.Report(pass, next.cc, fmt.Sprintf("unreachable case clause: %s will always match before %s", T.String(), V.String()),
						report.ShortRange())
				}
			}
		}
	}

	code.Preorder(pass, fn, (*ast.TypeSwitchStmt)(nil))
	return nil, nil
}

var checkSingleArgAppendQ = pattern.MustParse(`(CallExpr (Builtin "append") [_])`)

func CheckSingleArgAppend(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		_, ok := code.Match(pass, checkSingleArgAppendQ, node)
		if !ok {
			return
		}
		report.Report(pass, node, "x = append(y) is equivalent to x = y", report.FilterGenerated())
	}
	code.Preorder(pass, fn, (*ast.CallExpr)(nil))
	return nil, nil
}

func CheckStructTags(pass *analysis.Pass) (interface{}, error) {
	importsGoFlags := false

	// we use the AST instead of (*types.Package).Imports to work
	// around vendored packages in GOPATH mode. A vendored package's
	// path will include the vendoring subtree as a prefix.
	for _, f := range pass.Files {
		for _, imp := range f.Imports {
			v := imp.Path.Value
			if v[1:len(v)-1] == "github.com/jessevdk/go-flags" {
				importsGoFlags = true
				break
			}
		}
	}

	fn := func(node ast.Node) {
		structNode := node.(*ast.StructType)
		T := pass.TypesInfo.Types[structNode].Type.(*types.Struct)
		rt := fakereflect.TypeAndCanAddr{
			Type: T,
		}
		for i, field := range structNode.Fields.List {
			if field.Tag == nil {
				continue
			}
			tags, err := parseStructTag(field.Tag.Value[1 : len(field.Tag.Value)-1])
			if err != nil {
				report.Report(pass, field.Tag, fmt.Sprintf("unparseable struct tag: %s", err))
				continue
			}
			for k, v := range tags {
				if len(v) > 1 {
					isGoFlagsTag := importsGoFlags &&
						(k == "choice" || k == "optional-value" || k == "default")
					if !isGoFlagsTag {
						report.Report(pass, field.Tag, fmt.Sprintf("duplicate struct tag %q", k))
					}
				}

				switch k {
				case "json":
					checkJSONTag(pass, field, v[0])
				case "xml":
					if _, err := fakexml.StructFieldInfo(rt.Field(i)); err != nil {
						report.Report(pass, field.Tag, fmt.Sprintf("invalid XML tag: %s", err))
					}
					checkXMLTag(pass, field, v[0])
				}
			}
		}
	}
	code.Preorder(pass, fn, (*ast.StructType)(nil))
	return nil, nil
}

func checkJSONTag(pass *analysis.Pass, field *ast.Field, tag string) {
	if pass.Pkg.Path() == "encoding/json" || pass.Pkg.Path() == "encoding/json_test" {
		// don't flag malformed JSON tags in the encoding/json
		// package; it knows what it is doing, and it is testing
		// itself.
		return
	}
	//lint:ignore SA9003 TODO(dh): should we flag empty tags?
	if len(tag) == 0 {
	}
	fields := strings.Split(tag, ",")
	for _, r := range fields[0] {
		if !unicode.IsLetter(r) && !unicode.IsDigit(r) && !strings.ContainsRune("!#$%&()*+-./:<=>?@[]^_{|}~ ", r) {
			report.Report(pass, field.Tag, fmt.Sprintf("invalid JSON field name %q", fields[0]))
		}
	}
	var co, cs, ci int
	for _, s := range fields[1:] {
		switch s {
		case "omitempty":
			co++
		case "":
			// allow stuff like "-,"
		case "string":
			cs++
			// only for string, floating point, integer and bool
			T := typeutil.Dereference(pass.TypesInfo.TypeOf(field.Type).Underlying()).Underlying()
			basic, ok := T.(*types.Basic)
			if !ok || (basic.Info()&(types.IsBoolean|types.IsInteger|types.IsFloat|types.IsString)) == 0 {
				report.Report(pass, field.Tag, "the JSON string option only applies to fields of type string, floating point, integer or bool, or pointers to those")
			}
		case "inline":
			ci++
		default:
			report.Report(pass, field.Tag, fmt.Sprintf("unknown JSON option %q", s))
		}
	}
	if co > 1 {
		report.Report(pass, field.Tag, `duplicate JSON option "omitempty"`)
	}
	if cs > 1 {
		report.Report(pass, field.Tag, `duplicate JSON option "string"`)
	}
	if ci > 1 {
		report.Report(pass, field.Tag, `duplicate JSON option "inline"`)
	}
}

func checkXMLTag(pass *analysis.Pass, field *ast.Field, tag string) {
	//lint:ignore SA9003 TODO(dh): should we flag empty tags?
	if len(tag) == 0 {
	}
	fields := strings.Split(tag, ",")
	counts := map[string]int{}
	for _, s := range fields[1:] {
		switch s {
		case "attr", "chardata", "cdata", "innerxml", "comment":
			counts[s]++
		case "omitempty", "any":
			counts[s]++
		case "":
		default:
			report.Report(pass, field.Tag, fmt.Sprintf("invalid XML tag: unknown option %q", s))
		}
	}
	for k, v := range counts {
		if v > 1 {
			report.Report(pass, field.Tag, fmt.Sprintf("invalid XML tag: duplicate option %q", k))
		}
	}
}

func CheckImpossibleTypeAssertion(pass *analysis.Pass) (interface{}, error) {
	type entry struct {
		l, r *types.Func
	}

	msc := &pass.ResultOf[buildir.Analyzer].(*buildir.IR).Pkg.Prog.MethodSets
	for _, fn := range pass.ResultOf[buildir.Analyzer].(*buildir.IR).SrcFuncs {
		for _, b := range fn.Blocks {
			for _, instr := range b.Instrs {
				assert, ok := instr.(*ir.TypeAssert)
				if !ok {
					continue
				}
				var wrong []entry
				left := assert.X.Type()
				right := assert.AssertedType
				righti, ok := right.Underlying().(*types.Interface)

				if !ok {
					// We only care about interface->interface
					// assertions. The Go compiler already catches
					// impossible interface->concrete assertions.
					continue
				}

				ms := msc.MethodSet(left)
				for i := 0; i < righti.NumMethods(); i++ {
					mr := righti.Method(i)
					sel := ms.Lookup(mr.Pkg(), mr.Name())
					if sel == nil {
						continue
					}
					ml := sel.Obj().(*types.Func)
					if types.AssignableTo(ml.Type(), mr.Type()) {
						continue
					}

					wrong = append(wrong, entry{ml, mr})
				}

				if len(wrong) != 0 {
					s := fmt.Sprintf("impossible type assertion; %s and %s contradict each other:",
						types.TypeString(left, types.RelativeTo(pass.Pkg)),
						types.TypeString(right, types.RelativeTo(pass.Pkg)))
					for _, e := range wrong {
						s += fmt.Sprintf("\n\twrong type for %s method", e.l.Name())
						s += fmt.Sprintf("\n\t\thave %s", e.l.Type())
						s += fmt.Sprintf("\n\t\twant %s", e.r.Type())
					}
					report.Report(pass, assert, s)
				}
			}
		}
	}
	return nil, nil
}

func checkWithValueKey(call *Call) {
	arg := call.Args[1]
	T := arg.Value.Value.Type()
	if T, ok := T.(*types.Basic); ok {
		arg.Invalid(
			fmt.Sprintf("should not use built-in type %s as key for value; define your own type to avoid collisions", T))
	}
	if !types.Comparable(T) {
		arg.Invalid(fmt.Sprintf("keys used with context.WithValue must be comparable, but type %s is not comparable", T))
	}
}

func CheckMaybeNil(pass *analysis.Pass) (interface{}, error) {
	// This is an extremely trivial check that doesn't try to reason
	// about control flow. That is, phis and sigmas do not propagate
	// any information. As such, we can flag this:
	//
	// 	_ = *x
	// 	if x == nil { return }
	//
	// but we cannot flag this:
	//
	// 	if x == nil { println(x) }
	// 	_ = *x
	//
	// nor many other variations of conditional uses of or assignments to x.
	//
	// However, even this trivial implementation finds plenty of
	// real-world bugs, such as dereference before nil pointer check,
	// or using t.Error instead of t.Fatal when encountering nil
	// pointers.
	//
	// On the flip side, our naive implementation avoids false positives in branches, such as
	//
	// 	if x != nil { _ = *x }
	//
	// due to the same lack of propagating information through sigma
	// nodes. x inside the branch will be independent of the x in the
	// nil pointer check.
	//
	//
	// We could implement a more powerful check, but then we'd be
	// getting false positives instead of false negatives because
	// we're incapable of deducing relationships between variables.
	// For example, a function might return a pointer and an error,
	// and the error being nil guarantees that the pointer is not nil.
	// Depending on the surrounding code, the pointer may still end up
	// being checked against nil in one place, and guarded by a check
	// on the error in another, which would lead to us marking some
	// loads as unsafe.
	//
	// Unfortunately, simply hard-coding the relationship between
	// return values wouldn't eliminate all false positives, either.
	// Many other more subtle relationships exist. An abridged example
	// from real code:
	//
	// if a == nil && b == nil { return }
	// c := fn(a)
	// if c != "" { _ = *a }
	//
	// where `fn` is guaranteed to return a non-empty string if a
	// isn't nil.
	//
	// We choose to err on the side of false negatives.

	isNilConst := func(v ir.Value) bool {
		if typeutil.IsPointerLike(v.Type()) {
			if k, ok := v.(*ir.Const); ok {
				return k.IsNil()
			}
		}
		return false
	}

	for _, fn := range pass.ResultOf[buildir.Analyzer].(*buildir.IR).SrcFuncs {
		maybeNil := map[ir.Value]ir.Instruction{}
		for _, b := range fn.Blocks {
			for _, instr := range b.Instrs {
				// Originally we looked at all ir.BinOp, but that would lead to calls like 'assert(x != nil)' causing false positives.
				// Restrict ourselves to actual if statements, as these are more likely to affect control flow in a way we can observe.
				if instr, ok := instr.(*ir.If); ok {
					if cond, ok := instr.Cond.(*ir.BinOp); ok {
						var ptr ir.Value
						if isNilConst(cond.X) {
							ptr = cond.Y
						} else if isNilConst(cond.Y) {
							ptr = cond.X
						}
						maybeNil[ptr] = cond
					}
				}
			}
		}

		for _, b := range fn.Blocks {
			for _, instr := range b.Instrs {
				var ptr ir.Value
				switch instr := instr.(type) {
				case *ir.Load:
					ptr = instr.X
				case *ir.Store:
					ptr = instr.Addr
				case *ir.IndexAddr:
					ptr = instr.X
					if _, ok := ptr.Type().Underlying().(*types.Slice); ok {
						// indexing a nil slice does not cause a nil pointer panic
						//
						// Note: This also works around the bad lowering of range loops over slices
						// (https://github.com/dominikh/go-tools/issues/1053)
						continue
					}
				case *ir.FieldAddr:
					ptr = instr.X
				}
				if ptr != nil {
					switch ptr.(type) {
					case *ir.Alloc, *ir.FieldAddr, *ir.IndexAddr:
						// these cannot be nil
						continue
					}
					if r, ok := maybeNil[ptr]; ok {
						report.Report(pass, instr, "possible nil pointer dereference",
							report.Related(r, "this check suggests that the pointer can be nil"))
					}
				}
			}
		}
	}

	return nil, nil
}

var checkAddressIsNilQ = pattern.MustParse(
	`(BinaryExpr
		(UnaryExpr "&" _)
		(Or "==" "!=")
		(Builtin "nil"))`)

func CheckAddressIsNil(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		_, ok := code.Match(pass, checkAddressIsNilQ, node)
		if !ok {
			return
		}
		report.Report(pass, node, "the address of a variable cannot be nil")
	}
	code.Preorder(pass, fn, (*ast.BinaryExpr)(nil))
	return nil, nil
}

var (
	checkFixedLengthTypeShiftQ = pattern.MustParse(`
		(Or
			(AssignStmt _ (Or ">>=" "<<=") _)
			(BinaryExpr _ (Or ">>" "<<") _))
	`)
)

func CheckStaticBitShift(pass *analysis.Pass) (interface{}, error) {
	isDubiousShift := func(x, y ast.Expr) (int64, int64, bool) {
		typ, ok := pass.TypesInfo.TypeOf(x).Underlying().(*types.Basic)
		if !ok {
			return 0, 0, false
		}
		switch typ.Kind() {
		case types.Int8, types.Int16, types.Int32, types.Int64,
			types.Uint8, types.Uint16, types.Uint32, types.Uint64:
			// We're only interested in fixed–size types.
		default:
			return 0, 0, false
		}

		const bitsInByte = 8
		typeBits := pass.TypesSizes.Sizeof(typ) * bitsInByte

		shiftLength, ok := code.ExprToInt(pass, y)
		if !ok {
			return 0, 0, false
		}

		return typeBits, shiftLength, shiftLength >= typeBits
	}

	fn := func(node ast.Node) {
		if _, ok := code.Match(pass, checkFixedLengthTypeShiftQ, node); !ok {
			return
		}

		switch e := node.(type) {
		case *ast.AssignStmt:
			if size, shift, yes := isDubiousShift(e.Lhs[0], e.Rhs[0]); yes {
				report.Report(pass, e, fmt.Sprintf("shifting %d-bit value by %d bits will always clear it", size, shift))
			}
		case *ast.BinaryExpr:
			if size, shift, yes := isDubiousShift(e.X, e.Y); yes {
				report.Report(pass, e, fmt.Sprintf("shifting %d-bit value by %d bits will always clear it", size, shift))
			}
		}
	}
	code.Preorder(pass, fn, (*ast.AssignStmt)(nil), (*ast.BinaryExpr)(nil))

	return nil, nil
}

func findSliceLenChecks(pass *analysis.Pass) {
	// mark all function parameters that have to be of even length
	for _, fn := range pass.ResultOf[buildir.Analyzer].(*buildir.IR).SrcFuncs {
		for _, b := range fn.Blocks {
			// all paths go through this block
			if !b.Dominates(fn.Exit) {
				continue
			}

			// if foo % 2 != 0
			ifi, ok := b.Control().(*ir.If)
			if !ok {
				continue
			}
			cmp, ok := ifi.Cond.(*ir.BinOp)
			if !ok {
				continue
			}
			var needle uint64
			switch cmp.Op {
			case token.NEQ:
				// look for != 0
				needle = 0
			case token.EQL:
				// look for == 1
				needle = 1
			default:
				continue
			}

			rem, ok1 := cmp.X.(*ir.BinOp)
			k, ok2 := cmp.Y.(*ir.Const)
			if ok1 != ok2 {
				continue
			}
			if !ok1 {
				rem, ok1 = cmp.Y.(*ir.BinOp)
				k, ok2 = cmp.X.(*ir.Const)
			}
			if !ok1 || !ok2 || rem.Op != token.REM || k.Value.Kind() != constant.Int || k.Uint64() != needle {
				continue
			}
			k, ok = rem.Y.(*ir.Const)
			if !ok || k.Value.Kind() != constant.Int || k.Uint64() != 2 {
				continue
			}

			// if len(foo) % 2 != 0
			call, ok := rem.X.(*ir.Call)
			if !ok || !irutil.IsCallTo(call.Common(), "len") {
				continue
			}

			// we're checking the length of a parameter that is a slice
			// TODO(dh): support parameters that have flown through sigmas and phis
			param, ok := call.Call.Args[0].(*ir.Parameter)
			if !ok {
				continue
			}
			if _, ok := param.Type().Underlying().(*types.Slice); !ok {
				continue
			}

			// if len(foo) % 2 != 0 then panic
			if _, ok := b.Succs[0].Control().(*ir.Panic); !ok {
				continue
			}

			pass.ExportObjectFact(param.Object(), new(evenElements))
		}
	}
}

func findIndirectSliceLenChecks(pass *analysis.Pass) {
	seen := map[*ir.Function]struct{}{}

	var doFunction func(fn *ir.Function)
	doFunction = func(fn *ir.Function) {
		if _, ok := seen[fn]; ok {
			return
		}
		seen[fn] = struct{}{}

		for _, b := range fn.Blocks {
			// all paths go through this block
			if !b.Dominates(fn.Exit) {
				continue
			}

			for _, instr := range b.Instrs {
				call, ok := instr.(*ir.Call)
				if !ok {
					continue
				}
				callee := call.Call.StaticCallee()
				if callee == nil {
					continue
				}

				if callee.Pkg == fn.Pkg {
					// TODO(dh): are we missing interesting wrappers
					// because wrappers don't have Pkg set?
					doFunction(callee)
				}

				for argi, arg := range call.Call.Args {
					if callee.Signature.Recv() != nil {
						if argi == 0 {
							continue
						}
						argi--
					}

					// TODO(dh): support parameters that have flown through sigmas and phis
					param, ok := arg.(*ir.Parameter)
					if !ok {
						continue
					}
					if _, ok := param.Type().Underlying().(*types.Slice); !ok {
						continue
					}

					// We can't use callee.Params to look up the
					// parameter, because Params is not populated for
					// external functions. In our modular analysis.
					// any function in any package that isn't the
					// current package is considered "external", as it
					// has been loaded from export data only.
					sigParams := callee.Signature.Params()

					if !pass.ImportObjectFact(sigParams.At(argi), new(evenElements)) {
						continue
					}
					pass.ExportObjectFact(param.Object(), new(evenElements))
				}
			}
		}
	}

	for _, fn := range pass.ResultOf[buildir.Analyzer].(*buildir.IR).SrcFuncs {
		doFunction(fn)
	}
}

func findSliceLength(v ir.Value) int {
	// TODO(dh): VRP would help here

	v = irutil.Flatten(v)
	val := func(v ir.Value) int {
		if v, ok := v.(*ir.Const); ok {
			return int(v.Int64())
		}
		return -1
	}
	switch v := v.(type) {
	case *ir.Slice:
		low := 0
		high := -1
		if v.Low != nil {
			low = val(v.Low)
		}
		if v.High != nil {
			high = val(v.High)
		} else {
			switch vv := v.X.(type) {
			case *ir.Alloc:
				high = int(typeutil.Dereference(vv.Type()).Underlying().(*types.Array).Len())
			case *ir.Slice:
				high = findSliceLength(vv)
			}
		}
		if low == -1 || high == -1 {
			return -1
		}
		return high - low
	default:
		return -1
	}
}

type evenElements struct{}

func (evenElements) AFact() {}

func (evenElements) String() string { return "needs even elements" }

func flagSliceLens(pass *analysis.Pass) {
	var tag evenElements

	for _, fn := range pass.ResultOf[buildir.Analyzer].(*buildir.IR).SrcFuncs {
		for _, b := range fn.Blocks {
			for _, instr := range b.Instrs {
				call, ok := instr.(ir.CallInstruction)
				if !ok {
					continue
				}
				callee := call.Common().StaticCallee()
				if callee == nil {
					continue
				}
				for argi, arg := range call.Common().Args {
					if callee.Signature.Recv() != nil {
						if argi == 0 {
							continue
						}
						argi--
					}

					_, ok := arg.Type().Underlying().(*types.Slice)
					if !ok {
						continue
					}
					param := callee.Signature.Params().At(argi)
					if !pass.ImportObjectFact(param, &tag) {
						continue
					}

					// TODO handle stubs

					// we know the argument has to have even length.
					// now let's try to find its length
					if n := findSliceLength(arg); n > -1 && n%2 != 0 {
						src := call.Source().(*ast.CallExpr).Args[argi]
						sig := call.Common().Signature()
						var label string
						if argi == sig.Params().Len()-1 && sig.Variadic() {
							label = "variadic argument"
						} else {
							label = "argument"
						}
						// Note that param.Name() is guaranteed to not
						// be empty, otherwise the function couldn't
						// have enforced its length.
						report.Report(pass, src, fmt.Sprintf("%s %q is expected to have even number of elements, but has %d elements", label, param.Name(), n))
					}
				}
			}
		}
	}
}

func CheckEvenSliceLength(pass *analysis.Pass) (interface{}, error) {
	findSliceLenChecks(pass)
	findIndirectSliceLenChecks(pass)
	flagSliceLens(pass)

	return nil, nil
}

func CheckTypedNilInterface(pass *analysis.Pass) (interface{}, error) {
	// The comparison 'fn() == nil' can never be true if fn() returns
	// an interface value and only returns typed nils. This is usually
	// a mistake in the function itself, but all we can say for
	// certain is that the comparison is pointless.
	//
	// Flag results if no untyped nils are being returned, but either
	// known typed nils, or typed unknown nilness are being returned.

	irpkg := pass.ResultOf[buildir.Analyzer].(*buildir.IR)
	typedness := pass.ResultOf[typedness.Analysis].(*typedness.Result)
	nilness := pass.ResultOf[nilness.Analysis].(*nilness.Result)
	for _, fn := range irpkg.SrcFuncs {
		for _, b := range fn.Blocks {
			for _, instr := range b.Instrs {
				binop, ok := instr.(*ir.BinOp)
				if !ok || !(binop.Op == token.EQL || binop.Op == token.NEQ) {
					continue
				}
				if _, ok := binop.X.Type().Underlying().(*types.Interface); !ok {
					// TODO support swapped X and Y
					continue
				}

				k, ok := binop.Y.(*ir.Const)
				if !ok || !k.IsNil() {
					// if binop.X is an interface, then binop.Y can
					// only be a Const if its untyped. A typed nil
					// constant would first be passed to
					// MakeInterface.
					continue
				}

				var idx int
				var obj *types.Func
				switch x := irutil.Flatten(binop.X).(type) {
				case *ir.Call:
					callee := x.Call.StaticCallee()
					if callee == nil {
						continue
					}
					obj, _ = callee.Object().(*types.Func)
					idx = 0
				case *ir.Extract:
					call, ok := irutil.Flatten(x.Tuple).(*ir.Call)
					if !ok {
						continue
					}
					callee := call.Call.StaticCallee()
					if callee == nil {
						continue
					}
					obj, _ = callee.Object().(*types.Func)
					idx = x.Index
				case *ir.MakeInterface:
					var qualifier string
					switch binop.Op {
					case token.EQL:
						qualifier = "never"
					case token.NEQ:
						qualifier = "always"
					default:
						panic("unreachable")
					}
					if report.HasRange(x.X) {
						report.Report(pass, binop, fmt.Sprintf("this comparison is %s true", qualifier),
							report.Related(x.X, "the lhs of the comparison gets its value from here and has a concrete type"))
					} else {
						// we can't generate related information for this, so make the diagnostic itself slightly more useful
						report.Report(pass, binop, fmt.Sprintf("this comparison is %s true; the lhs of the comparison has been assigned a concretely typed value", qualifier))
					}
					continue
				}
				if obj == nil {
					continue
				}

				isNil, onlyGlobal := nilness.MayReturnNil(obj, idx)
				if typedness.MustReturnTyped(obj, idx) && isNil && !onlyGlobal && !code.IsInTest(pass, binop) {
					// Don't flag these comparisons in tests. Tests
					// may be explicitly enforcing the invariant that
					// a value isn't nil.

					var qualifier string
					switch binop.Op {
					case token.EQL:
						qualifier = "never"
					case token.NEQ:
						qualifier = "always"
					default:
						panic("unreachable")
					}
					report.Report(pass, binop, fmt.Sprintf("this comparison is %s true", qualifier),
						// TODO support swapped X and Y
						report.Related(binop.X, fmt.Sprintf("the lhs of the comparison is the %s return value of this function call", report.Ordinal(idx+1))),
						report.Related(obj, fmt.Sprintf("%s never returns a nil interface value", typeutil.FuncName(obj))))
				}
			}
		}
	}

	return nil, nil
}

var builtinLessThanZeroQ = pattern.MustParse(`
	(Or
		(BinaryExpr
			(BasicLit "INT" "0")
			">"
			(CallExpr builtin@(Builtin (Or "len" "cap")) _))
		(BinaryExpr
			(CallExpr builtin@(Builtin (Or "len" "cap")) _)
			"<"
			(BasicLit "INT" "0")))
`)

func CheckBuiltinZeroComparison(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		matcher, ok := code.Match(pass, builtinLessThanZeroQ, node)
		if !ok {
			return
		}

		builtin := matcher.State["builtin"].(*ast.Ident)
		report.Report(pass, node, fmt.Sprintf("builtin function %s does not return negative values", builtin.Name))
	}
	code.Preorder(pass, fn, (*ast.BinaryExpr)(nil))

	return nil, nil
}

var integerDivisionQ = pattern.MustParse(`(BinaryExpr (BasicLit "INT" _) "/" (BasicLit "INT" _))`)

func CheckIntegerDivisionEqualsZero(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		_, ok := code.Match(pass, integerDivisionQ, node)
		if !ok {
			return
		}

		val := constant.ToInt(pass.TypesInfo.Types[node.(ast.Expr)].Value)
		if v, ok := constant.Uint64Val(val); ok && v == 0 {
			report.Report(pass, node, fmt.Sprintf("the integer division '%s' results in zero", report.Render(pass, node)))
		}

		// TODO: we could offer a suggested fix here, but I am not
		// sure what it should be. There are many options to choose
		// from.

		// Note: we experimented with flagging divisions that truncate
		// (e.g. 4 / 3), but it ran into false positives in Go's
		// 'time' package, which does this, deliberately:
		//
		//   unixToInternal int64 = (1969*365 + 1969/4 - 1969/100 + 1969/400) * secondsPerDay
		//
		// The check also found a real bug in other code, but I don't
		// think we can outright ban this kind of division.
	}
	code.Preorder(pass, fn, (*ast.BinaryExpr)(nil))

	return nil, nil
}

func CheckIneffectiveFieldAssignments(pass *analysis.Pass) (interface{}, error) {
	// The analysis only considers the receiver and its first level
	// fields. It doesn't look at other parameters, nor at nested
	// fields.
	//
	// The analysis does not detect all kinds of dead stores, only
	// those of fields that are never read after the write. That is,
	// we do not flag 'a.x = 1; a.x = 2; _ = a.x'. We might explore
	// this again if we add support for SROA to go/ir and implement
	// https://github.com/dominikh/go-tools/issues/191.

	irpkg := pass.ResultOf[buildir.Analyzer].(*buildir.IR)
fnLoop:
	for _, fn := range irpkg.SrcFuncs {
		if recv := fn.Signature.Recv(); recv == nil {
			continue
		} else if _, ok := recv.Type().Underlying().(*types.Struct); !ok {
			continue
		}

		recv := fn.Params[0]
		refs := irutil.FilterDebug(*recv.Referrers())
		if len(refs) != 1 {
			continue
		}
		store, ok := refs[0].(*ir.Store)
		if !ok {
			continue
		}
		alloc, ok := store.Addr.(*ir.Alloc)
		if !ok || alloc.Heap {
			continue
		}

		reads := map[int][]ir.Instruction{}
		writes := map[int][]ir.Instruction{}
		for _, ref := range *alloc.Referrers() {
			switch ref := ref.(type) {
			case *ir.FieldAddr:
				for _, refref := range *ref.Referrers() {
					switch refref.(type) {
					case *ir.Store:
						writes[ref.Field] = append(writes[ref.Field], refref)
					case *ir.Load:
						reads[ref.Field] = append(reads[ref.Field], refref)
					case *ir.DebugRef:
						continue
					default:
						// this should be safe… if the field address
						// escapes, then alloc.Heap will be true.
						// there should be no instructions left that,
						// given this FieldAddr, without escaping, can
						// effect a load or store.
						continue
					}
				}
			case *ir.Store:
				// we could treat this as a store to every field, but
				// we don't want to decide the semantics of partial
				// struct initializers. should `v = t{x: 1}` also mark
				// v.y as being written to?
				if ref != store {
					continue fnLoop
				}
			case *ir.Load:
				// a load of the entire struct loads every field
				for i := 0; i < recv.Type().Underlying().(*types.Struct).NumFields(); i++ {
					reads[i] = append(reads[i], ref)
				}
			case *ir.DebugRef:
				continue
			default:
				continue fnLoop
			}
		}

		offset := func(instr ir.Instruction) int {
			for i, other := range instr.Block().Instrs {
				if instr == other {
					return i
				}
			}
			panic("couldn't find instruction in its block")
		}

		for field, ws := range writes {
			rs := reads[field]
		wLoop:
			for _, w := range ws {
				for _, r := range rs {
					if w.Block() == r.Block() {
						if offset(r) > offset(w) {
							// found a reachable read of our write
							continue wLoop
						}
					} else if irutil.Reachable(w.Block(), r.Block()) {
						// found a reachable read of our write
						continue wLoop
					}
				}
				fieldName := recv.Type().Underlying().(*types.Struct).Field(field).Name()
				report.Report(pass, w, fmt.Sprintf("ineffective assignment to field %s.%s", recv.Type().(*types.Named).Obj().Name(), fieldName))
			}
		}
	}
	return nil, nil
}

var negativeZeroFloatQ = pattern.MustParse(`
	(Or
		(UnaryExpr
			"-"
			(BasicLit "FLOAT" "0.0"))

		(UnaryExpr
			"-"
			(CallExpr conv@(Object (Or "float32" "float64")) lit@(Or (BasicLit "INT" "0") (BasicLit "FLOAT" "0.0"))))

		(CallExpr
			conv@(Object (Or "float32" "float64"))
			(UnaryExpr "-" lit@(BasicLit "INT" "0"))))`)

func CheckNegativeZeroFloat(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		m, ok := code.Match(pass, negativeZeroFloatQ, node)
		if !ok {
			return
		}

		if conv, ok := m.State["conv"].(*types.TypeName); ok {
			var replacement string
			// TODO(dh): how does this handle type aliases?
			if conv.Name() == "float32" {
				replacement = `float32(math.Copysign(0, -1))`
			} else {
				replacement = `math.Copysign(0, -1)`
			}
			report.Report(pass, node,
				fmt.Sprintf("in Go, the floating-point expression '%s' is the same as '%s(%s)', it does not produce a negative zero",
					report.Render(pass, node),
					conv.Name(),
					report.Render(pass, m.State["lit"])),
				report.Fixes(edit.Fix("use math.Copysign to create negative zero", edit.ReplaceWithString(pass.Fset, node, replacement))))
		} else {
			const replacement = `math.Copysign(0, -1)`
			report.Report(pass, node,
				"in Go, the floating-point literal '-0.0' is the same as '0.0', it does not produce a negative zero",
				report.Fixes(edit.Fix("use math.Copysign to create negative zero", edit.ReplaceWithString(pass.Fset, node, replacement))))
		}
	}
	code.Preorder(pass, fn, (*ast.UnaryExpr)(nil), (*ast.CallExpr)(nil))
	return nil, nil
}

var ineffectiveURLQueryAddQ = pattern.MustParse(`(CallExpr (SelectorExpr (CallExpr (SelectorExpr recv (Ident "Query")) []) (Ident meth)) _)`)

func CheckIneffectiveURLQueryModification(pass *analysis.Pass) (interface{}, error) {
	// TODO(dh): We could make this check more complex and detect
	// pointless modifications of net/url.Values in general, but that
	// requires us to get the state machine correct, else we'll cause
	// false positives.

	fn := func(node ast.Node) {
		m, ok := code.Match(pass, ineffectiveURLQueryAddQ, node)
		if !ok {
			return
		}
		if !code.IsOfType(pass, m.State["recv"].(ast.Expr), "*net/url.URL") {
			return
		}
		switch m.State["meth"].(string) {
		case "Add", "Del", "Set":
		default:
			return
		}
		report.Report(pass, node, "(*net/url.URL).Query returns a copy, modifying it doesn't change the URL")
	}
	code.Preorder(pass, fn, (*ast.CallExpr)(nil))
	return nil, nil
}
