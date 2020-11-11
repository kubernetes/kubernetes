package staticcheck

import (
	"fmt"
	"go/constant"
	"go/types"
	"net"
	"net/url"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"golang.org/x/tools/go/analysis"
	"honnef.co/go/tools/code"
	"honnef.co/go/tools/ir"
)

const (
	MsgInvalidHostPort = "invalid port or service name in host:port pair"
	MsgInvalidUTF8     = "argument is not a valid UTF-8 encoded string"
	MsgNonUniqueCutset = "cutset contains duplicate characters"
)

type Call struct {
	Pass  *analysis.Pass
	Instr ir.CallInstruction
	Args  []*Argument

	Parent *ir.Function

	invalids []string
}

func (c *Call) Invalid(msg string) {
	c.invalids = append(c.invalids, msg)
}

type Argument struct {
	Value    Value
	invalids []string
}

type Value struct {
	Value ir.Value
}

func (arg *Argument) Invalid(msg string) {
	arg.invalids = append(arg.invalids, msg)
}

type CallCheck func(call *Call)

func extractConsts(v ir.Value) []*ir.Const {
	switch v := v.(type) {
	case *ir.Const:
		return []*ir.Const{v}
	case *ir.MakeInterface:
		return extractConsts(v.X)
	default:
		return nil
	}
}

func ValidateRegexp(v Value) error {
	for _, c := range extractConsts(v.Value) {
		if c.Value == nil {
			continue
		}
		if c.Value.Kind() != constant.String {
			continue
		}
		s := constant.StringVal(c.Value)
		if _, err := regexp.Compile(s); err != nil {
			return err
		}
	}
	return nil
}

func ValidateTimeLayout(v Value) error {
	for _, c := range extractConsts(v.Value) {
		if c.Value == nil {
			continue
		}
		if c.Value.Kind() != constant.String {
			continue
		}
		s := constant.StringVal(c.Value)
		s = strings.Replace(s, "_", " ", -1)
		s = strings.Replace(s, "Z", "-", -1)
		_, err := time.Parse(s, s)
		if err != nil {
			return err
		}
	}
	return nil
}

func ValidateURL(v Value) error {
	for _, c := range extractConsts(v.Value) {
		if c.Value == nil {
			continue
		}
		if c.Value.Kind() != constant.String {
			continue
		}
		s := constant.StringVal(c.Value)
		_, err := url.Parse(s)
		if err != nil {
			return fmt.Errorf("%q is not a valid URL: %s", s, err)
		}
	}
	return nil
}

func InvalidUTF8(v Value) bool {
	for _, c := range extractConsts(v.Value) {
		if c.Value == nil {
			continue
		}
		if c.Value.Kind() != constant.String {
			continue
		}
		s := constant.StringVal(c.Value)
		if !utf8.ValidString(s) {
			return true
		}
	}
	return false
}

func UnbufferedChannel(v Value) bool {
	// TODO(dh): this check of course misses many cases of unbuffered
	// channels, such as any in phi or sigma nodes. We'll eventually
	// replace this function.
	val := v.Value
	if ct, ok := val.(*ir.ChangeType); ok {
		val = ct.X
	}
	mk, ok := val.(*ir.MakeChan)
	if !ok {
		return false
	}
	if k, ok := mk.Size.(*ir.Const); ok && k.Value.Kind() == constant.Int {
		if v, ok := constant.Int64Val(k.Value); ok && v == 0 {
			return true
		}
	}
	return false
}

func Pointer(v Value) bool {
	switch v.Value.Type().Underlying().(type) {
	case *types.Pointer, *types.Interface:
		return true
	}
	return false
}

func ConvertedFromInt(v Value) bool {
	conv, ok := v.Value.(*ir.Convert)
	if !ok {
		return false
	}
	b, ok := conv.X.Type().Underlying().(*types.Basic)
	if !ok {
		return false
	}
	if (b.Info() & types.IsInteger) == 0 {
		return false
	}
	return true
}

func validEncodingBinaryType(pass *analysis.Pass, typ types.Type) bool {
	typ = typ.Underlying()
	switch typ := typ.(type) {
	case *types.Basic:
		switch typ.Kind() {
		case types.Uint8, types.Uint16, types.Uint32, types.Uint64,
			types.Int8, types.Int16, types.Int32, types.Int64,
			types.Float32, types.Float64, types.Complex64, types.Complex128, types.Invalid:
			return true
		case types.Bool:
			return code.IsGoVersion(pass, 8)
		}
		return false
	case *types.Struct:
		n := typ.NumFields()
		for i := 0; i < n; i++ {
			if !validEncodingBinaryType(pass, typ.Field(i).Type()) {
				return false
			}
		}
		return true
	case *types.Array:
		return validEncodingBinaryType(pass, typ.Elem())
	case *types.Interface:
		// we can't determine if it's a valid type or not
		return true
	}
	return false
}

func CanBinaryMarshal(pass *analysis.Pass, v Value) bool {
	typ := v.Value.Type().Underlying()
	if ttyp, ok := typ.(*types.Pointer); ok {
		typ = ttyp.Elem().Underlying()
	}
	if ttyp, ok := typ.(interface {
		Elem() types.Type
	}); ok {
		if _, ok := ttyp.(*types.Pointer); !ok {
			typ = ttyp.Elem()
		}
	}

	return validEncodingBinaryType(pass, typ)
}

func RepeatZeroTimes(name string, arg int) CallCheck {
	return func(call *Call) {
		arg := call.Args[arg]
		if k, ok := arg.Value.Value.(*ir.Const); ok && k.Value.Kind() == constant.Int {
			if v, ok := constant.Int64Val(k.Value); ok && v == 0 {
				arg.Invalid(fmt.Sprintf("calling %s with n == 0 will return no results, did you mean -1?", name))
			}
		}
	}
}

func validateServiceName(s string) bool {
	if len(s) < 1 || len(s) > 15 {
		return false
	}
	if s[0] == '-' || s[len(s)-1] == '-' {
		return false
	}
	if strings.Contains(s, "--") {
		return false
	}
	hasLetter := false
	for _, r := range s {
		if (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') {
			hasLetter = true
			continue
		}
		if r >= '0' && r <= '9' {
			continue
		}
		return false
	}
	return hasLetter
}

func validatePort(s string) bool {
	n, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return validateServiceName(s)
	}
	return n >= 0 && n <= 65535
}

func ValidHostPort(v Value) bool {
	for _, k := range extractConsts(v.Value) {
		if k.Value == nil {
			continue
		}
		if k.Value.Kind() != constant.String {
			continue
		}
		s := constant.StringVal(k.Value)
		_, port, err := net.SplitHostPort(s)
		if err != nil {
			return false
		}
		// TODO(dh): check hostname
		if !validatePort(port) {
			return false
		}
	}
	return true
}

// ConvertedFrom reports whether value v was converted from type typ.
func ConvertedFrom(v Value, typ string) bool {
	change, ok := v.Value.(*ir.ChangeType)
	return ok && code.IsType(change.X.Type(), typ)
}

func UniqueStringCutset(v Value) bool {
	for _, c := range extractConsts(v.Value) {
		if c.Value == nil {
			continue
		}
		if c.Value.Kind() != constant.String {
			continue
		}
		s := constant.StringVal(c.Value)
		rs := runeSlice(s)
		if len(rs) < 2 {
			continue
		}
		sort.Sort(rs)
		for i, r := range rs[1:] {
			if rs[i] == r {
				return false
			}
		}
	}
	return true
}
