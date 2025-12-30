package dbus

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"reflect"
	"strconv"
	"strings"
	"unicode/utf8"
)

type varParser struct {
	tokens []varToken
	i      int
}

func (p *varParser) backup() {
	p.i--
}

func (p *varParser) next() varToken {
	if p.i < len(p.tokens) {
		t := p.tokens[p.i]
		p.i++
		return t
	}
	return varToken{typ: tokEOF}
}

type varNode interface {
	Infer() (Signature, error)
	String() string
	Sigs() sigSet
	Value(Signature) (any, error)
}

func varMakeNode(p *varParser) (varNode, error) {
	var sig Signature

	for {
		t := p.next()
		switch t.typ {
		case tokEOF:
			return nil, io.ErrUnexpectedEOF
		case tokError:
			return nil, errors.New(t.val)
		case tokNumber:
			return varMakeNumNode(t, sig)
		case tokString:
			return varMakeStringNode(t, sig)
		case tokBool:
			if sig.str != "" && sig.str != "b" {
				return nil, varTypeError{t.val, sig}
			}
			b, err := strconv.ParseBool(t.val)
			if err != nil {
				return nil, err
			}
			return boolNode(b), nil
		case tokArrayStart:
			return varMakeArrayNode(p, sig)
		case tokVariantStart:
			return varMakeVariantNode(p, sig)
		case tokDictStart:
			return varMakeDictNode(p, sig)
		case tokType:
			if sig.str != "" {
				return nil, errors.New("unexpected type annotation")
			}
			if t.val[0] == '@' {
				sig.str = t.val[1:]
			} else {
				sig.str = varTypeMap[t.val]
			}
		case tokByteString:
			if sig.str != "" && sig.str != "ay" {
				return nil, varTypeError{t.val, sig}
			}
			b, err := varParseByteString(t.val)
			if err != nil {
				return nil, err
			}
			return byteStringNode(b), nil
		default:
			return nil, fmt.Errorf("unexpected %q", t.val)
		}
	}
}

type varTypeError struct {
	val string
	sig Signature
}

func (e varTypeError) Error() string {
	return fmt.Sprintf("dbus: can't parse %q as type %q", e.val, e.sig.str)
}

type sigSet map[Signature]bool

func (s sigSet) Empty() bool {
	return len(s) == 0
}

func (s sigSet) Intersect(s2 sigSet) sigSet {
	r := make(sigSet)
	for k := range s {
		if s2[k] {
			r[k] = true
		}
	}
	return r
}

func (s sigSet) Single() (Signature, bool) {
	if len(s) == 1 {
		for k := range s {
			return k, true
		}
	}
	return Signature{}, false
}

func (s sigSet) ToArray() sigSet {
	r := make(sigSet, len(s))
	for k := range s {
		r[Signature{"a" + k.str}] = true
	}
	return r
}

type numNode struct {
	sig Signature
	str string
	val any
}

var numSigSet = sigSet{
	Signature{"y"}: true,
	Signature{"n"}: true,
	Signature{"q"}: true,
	Signature{"i"}: true,
	Signature{"u"}: true,
	Signature{"x"}: true,
	Signature{"t"}: true,
	Signature{"d"}: true,
}

func (n numNode) Infer() (Signature, error) {
	if strings.ContainsAny(n.str, ".e") {
		return Signature{"d"}, nil
	}
	return Signature{"i"}, nil
}

func (n numNode) String() string {
	return n.str
}

func (n numNode) Sigs() sigSet {
	if n.sig.str != "" {
		return sigSet{n.sig: true}
	}
	if strings.ContainsAny(n.str, ".e") {
		return sigSet{Signature{"d"}: true}
	}
	return numSigSet
}

func (n numNode) Value(sig Signature) (any, error) {
	if n.sig.str != "" && n.sig != sig {
		return nil, varTypeError{n.str, sig}
	}
	if n.val != nil {
		return n.val, nil
	}
	return varNumAs(n.str, sig)
}

func varMakeNumNode(tok varToken, sig Signature) (varNode, error) {
	if sig.str == "" {
		return numNode{str: tok.val}, nil
	}
	num, err := varNumAs(tok.val, sig)
	if err != nil {
		return nil, err
	}
	return numNode{sig: sig, val: num}, nil
}

func varNumAs(s string, sig Signature) (any, error) {
	isUnsigned := false
	size := 32
	switch sig.str {
	case "n":
		size = 16
	case "i":
	case "x":
		size = 64
	case "y":
		size = 8
		isUnsigned = true
	case "q":
		size = 16
		isUnsigned = true
	case "u":
		isUnsigned = true
	case "t":
		size = 64
		isUnsigned = true
	case "d":
		d, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return nil, err
		}
		return d, nil
	default:
		return nil, varTypeError{s, sig}
	}
	base := 10
	if after, ok := strings.CutPrefix(s, "0x"); ok {
		base = 16
		s = after
	}
	if after, ok := strings.CutPrefix(s, "0"); ok && len(s) != 1 {
		base = 8
		s = after
	}
	if isUnsigned {
		i, err := strconv.ParseUint(s, base, size)
		if err != nil {
			return nil, err
		}
		var v any = i
		switch sig.str {
		case "y":
			v = byte(i)
		case "q":
			v = uint16(i)
		case "u":
			v = uint32(i)
		}
		return v, nil
	}
	i, err := strconv.ParseInt(s, base, size)
	if err != nil {
		return nil, err
	}
	var v any = i
	switch sig.str {
	case "n":
		v = int16(i)
	case "i":
		v = int32(i)
	}
	return v, nil
}

type stringNode struct {
	sig Signature
	str string // parsed
	val any    // has correct type
}

var stringSigSet = sigSet{
	Signature{"s"}: true,
	Signature{"g"}: true,
	Signature{"o"}: true,
}

func (n stringNode) Infer() (Signature, error) {
	return Signature{"s"}, nil
}

func (n stringNode) String() string {
	return n.str
}

func (n stringNode) Sigs() sigSet {
	if n.sig.str != "" {
		return sigSet{n.sig: true}
	}
	return stringSigSet
}

func (n stringNode) Value(sig Signature) (any, error) {
	if n.sig.str != "" && n.sig != sig {
		return nil, varTypeError{n.str, sig}
	}
	if n.val != nil {
		return n.val, nil
	}
	switch {
	case sig.str == "g":
		return Signature{n.str}, nil
	case sig.str == "o":
		return ObjectPath(n.str), nil
	case sig.str == "s":
		return n.str, nil
	default:
		return nil, varTypeError{n.str, sig}
	}
}

func varMakeStringNode(tok varToken, sig Signature) (varNode, error) {
	if sig.str != "" && sig.str != "s" && sig.str != "g" && sig.str != "o" {
		return nil, fmt.Errorf("invalid type %q for string", sig.str)
	}
	s, err := varParseString(tok.val)
	if err != nil {
		return nil, err
	}
	n := stringNode{str: s}
	if sig.str == "" {
		return stringNode{str: s}, nil
	}
	n.sig = sig
	switch sig.str {
	case "o":
		n.val = ObjectPath(s)
	case "g":
		n.val = Signature{s}
	case "s":
		n.val = s
	}
	return n, nil
}

func varParseString(s string) (string, error) {
	// quotes are guaranteed to be there
	s = s[1 : len(s)-1]
	buf := new(bytes.Buffer)
	for len(s) != 0 {
		r, size := utf8.DecodeRuneInString(s)
		if r == utf8.RuneError && size == 1 {
			return "", errors.New("invalid UTF-8")
		}
		s = s[size:]
		if r != '\\' {
			buf.WriteRune(r)
			continue
		}
		r, size = utf8.DecodeRuneInString(s)
		if r == utf8.RuneError && size == 1 {
			return "", errors.New("invalid UTF-8")
		}
		s = s[size:]
		switch r {
		case 'a':
			buf.WriteRune(0x7)
		case 'b':
			buf.WriteRune(0x8)
		case 'f':
			buf.WriteRune(0xc)
		case 'n':
			buf.WriteRune('\n')
		case 'r':
			buf.WriteRune('\r')
		case 't':
			buf.WriteRune('\t')
		case '\n':
		case 'u':
			if len(s) < 4 {
				return "", errors.New("short unicode escape")
			}
			r, err := strconv.ParseUint(s[:4], 16, 32)
			if err != nil {
				return "", err
			}
			buf.WriteRune(rune(r))
			s = s[4:]
		case 'U':
			if len(s) < 8 {
				return "", errors.New("short unicode escape")
			}
			r, err := strconv.ParseUint(s[:8], 16, 32)
			if err != nil {
				return "", err
			}
			buf.WriteRune(rune(r))
			s = s[8:]
		default:
			buf.WriteRune(r)
		}
	}
	return buf.String(), nil
}

var boolSigSet = sigSet{Signature{"b"}: true}

type boolNode bool

func (boolNode) Infer() (Signature, error) {
	return Signature{"b"}, nil
}

func (b boolNode) String() string {
	if b {
		return "true"
	}
	return "false"
}

func (boolNode) Sigs() sigSet {
	return boolSigSet
}

func (b boolNode) Value(sig Signature) (any, error) {
	if sig.str != "b" {
		return nil, varTypeError{b.String(), sig}
	}
	return bool(b), nil
}

type arrayNode struct {
	set      sigSet
	children []varNode
}

func (n arrayNode) Infer() (Signature, error) {
	for _, v := range n.children {
		csig, err := varInfer(v)
		if err != nil {
			continue
		}
		return Signature{"a" + csig.str}, nil
	}
	return Signature{}, fmt.Errorf("can't infer type for %q", n.String())
}

func (n arrayNode) String() string {
	s := "["
	for i, v := range n.children {
		s += v.String()
		if i != len(n.children)-1 {
			s += ", "
		}
	}
	return s + "]"
}

func (n arrayNode) Sigs() sigSet {
	return n.set
}

func (n arrayNode) Value(sig Signature) (any, error) {
	if n.set.Empty() {
		// no type information whatsoever, so this must be an empty slice
		return reflect.MakeSlice(typeFor(sig.str), 0, 0).Interface(), nil
	}
	if !n.set[sig] {
		return nil, varTypeError{n.String(), sig}
	}
	s := reflect.MakeSlice(typeFor(sig.str), len(n.children), len(n.children))
	for i, v := range n.children {
		rv, err := v.Value(Signature{sig.str[1:]})
		if err != nil {
			return nil, err
		}
		s.Index(i).Set(reflect.ValueOf(rv))
	}
	return s.Interface(), nil
}

func varMakeArrayNode(p *varParser, sig Signature) (varNode, error) {
	var n arrayNode
	if sig.str != "" {
		n.set = sigSet{sig: true}
	}
	if t := p.next(); t.typ == tokArrayEnd {
		return n, nil
	} else {
		p.backup()
	}
Loop:
	for {
		t := p.next()
		switch t.typ {
		case tokEOF:
			return nil, io.ErrUnexpectedEOF
		case tokError:
			return nil, errors.New(t.val)
		}
		p.backup()
		cn, err := varMakeNode(p)
		if err != nil {
			return nil, err
		}
		if cset := cn.Sigs(); !cset.Empty() {
			if n.set.Empty() {
				n.set = cset.ToArray()
			} else {
				nset := cset.ToArray().Intersect(n.set)
				if nset.Empty() {
					return nil, fmt.Errorf("can't parse %q with given type information", cn.String())
				}
				n.set = nset
			}
		}
		n.children = append(n.children, cn)
		switch t := p.next(); t.typ {
		case tokEOF:
			return nil, io.ErrUnexpectedEOF
		case tokError:
			return nil, errors.New(t.val)
		case tokArrayEnd:
			break Loop
		case tokComma:
			continue
		default:
			return nil, fmt.Errorf("unexpected %q", t.val)
		}
	}
	return n, nil
}

type variantNode struct {
	n varNode
}

var variantSet = sigSet{
	Signature{"v"}: true,
}

func (variantNode) Infer() (Signature, error) {
	return Signature{"v"}, nil
}

func (n variantNode) String() string {
	return "<" + n.n.String() + ">"
}

func (variantNode) Sigs() sigSet {
	return variantSet
}

func (n variantNode) Value(sig Signature) (any, error) {
	if sig.str != "v" {
		return nil, varTypeError{n.String(), sig}
	}
	sig, err := varInfer(n.n)
	if err != nil {
		return nil, err
	}
	v, err := n.n.Value(sig)
	if err != nil {
		return nil, err
	}
	return MakeVariant(v), nil
}

func varMakeVariantNode(p *varParser, sig Signature) (varNode, error) {
	n, err := varMakeNode(p)
	if err != nil {
		return nil, err
	}
	if t := p.next(); t.typ != tokVariantEnd {
		return nil, fmt.Errorf("unexpected %q", t.val)
	}
	vn := variantNode{n}
	if sig.str != "" && sig.str != "v" {
		return nil, varTypeError{vn.String(), sig}
	}
	return variantNode{n}, nil
}

type dictEntry struct {
	key, val varNode
}

type dictNode struct {
	kset, vset sigSet
	children   []dictEntry
}

func (n dictNode) Infer() (Signature, error) {
	for _, v := range n.children {
		ksig, err := varInfer(v.key)
		if err != nil {
			continue
		}
		vsig, err := varInfer(v.val)
		if err != nil {
			continue
		}
		return Signature{"a{" + ksig.str + vsig.str + "}"}, nil
	}
	return Signature{}, fmt.Errorf("can't infer type for %q", n.String())
}

func (n dictNode) String() string {
	s := "{"
	for i, v := range n.children {
		s += v.key.String() + ": " + v.val.String()
		if i != len(n.children)-1 {
			s += ", "
		}
	}
	return s + "}"
}

func (n dictNode) Sigs() sigSet {
	r := sigSet{}
	for k := range n.kset {
		for v := range n.vset {
			sig := "a{" + k.str + v.str + "}"
			r[Signature{sig}] = true
		}
	}
	return r
}

func (n dictNode) Value(sig Signature) (any, error) {
	set := n.Sigs()
	if set.Empty() {
		// no type information -> empty dict
		return reflect.MakeMap(typeFor(sig.str)).Interface(), nil
	}
	if !set[sig] {
		return nil, varTypeError{n.String(), sig}
	}
	m := reflect.MakeMap(typeFor(sig.str))
	ksig := Signature{sig.str[2:3]}
	vsig := Signature{sig.str[3 : len(sig.str)-1]}
	for _, v := range n.children {
		kv, err := v.key.Value(ksig)
		if err != nil {
			return nil, err
		}
		vv, err := v.val.Value(vsig)
		if err != nil {
			return nil, err
		}
		m.SetMapIndex(reflect.ValueOf(kv), reflect.ValueOf(vv))
	}
	return m.Interface(), nil
}

func varMakeDictNode(p *varParser, sig Signature) (varNode, error) {
	var n dictNode

	if sig.str != "" {
		if len(sig.str) < 5 {
			return nil, fmt.Errorf("invalid signature %q for dict type", sig)
		}
		ksig := Signature{string(sig.str[2])}
		vsig := Signature{sig.str[3 : len(sig.str)-1]}
		n.kset = sigSet{ksig: true}
		n.vset = sigSet{vsig: true}
	}
	if t := p.next(); t.typ == tokDictEnd {
		return n, nil
	} else {
		p.backup()
	}
Loop:
	for {
		t := p.next()
		switch t.typ {
		case tokEOF:
			return nil, io.ErrUnexpectedEOF
		case tokError:
			return nil, errors.New(t.val)
		}
		p.backup()
		kn, err := varMakeNode(p)
		if err != nil {
			return nil, err
		}
		if kset := kn.Sigs(); !kset.Empty() {
			if n.kset.Empty() {
				n.kset = kset
			} else {
				n.kset = kset.Intersect(n.kset)
				if n.kset.Empty() {
					return nil, fmt.Errorf("can't parse %q with given type information", kn.String())
				}
			}
		}
		t = p.next()
		switch t.typ {
		case tokEOF:
			return nil, io.ErrUnexpectedEOF
		case tokError:
			return nil, errors.New(t.val)
		case tokColon:
		default:
			return nil, fmt.Errorf("unexpected %q", t.val)
		}
		t = p.next()
		switch t.typ {
		case tokEOF:
			return nil, io.ErrUnexpectedEOF
		case tokError:
			return nil, errors.New(t.val)
		}
		p.backup()
		vn, err := varMakeNode(p)
		if err != nil {
			return nil, err
		}
		if vset := vn.Sigs(); !vset.Empty() {
			if n.vset.Empty() {
				n.vset = vset
			} else {
				n.vset = n.vset.Intersect(vset)
				if n.vset.Empty() {
					return nil, fmt.Errorf("can't parse %q with given type information", vn.String())
				}
			}
		}
		n.children = append(n.children, dictEntry{kn, vn})
		t = p.next()
		switch t.typ {
		case tokEOF:
			return nil, io.ErrUnexpectedEOF
		case tokError:
			return nil, errors.New(t.val)
		case tokDictEnd:
			break Loop
		case tokComma:
			continue
		default:
			return nil, fmt.Errorf("unexpected %q", t.val)
		}
	}
	return n, nil
}

type byteStringNode []byte

var byteStringSet = sigSet{
	Signature{"ay"}: true,
}

func (byteStringNode) Infer() (Signature, error) {
	return Signature{"ay"}, nil
}

func (b byteStringNode) String() string {
	return string(b)
}

func (b byteStringNode) Sigs() sigSet {
	return byteStringSet
}

func (b byteStringNode) Value(sig Signature) (any, error) {
	if sig.str != "ay" {
		return nil, varTypeError{b.String(), sig}
	}
	return []byte(b), nil
}

func varParseByteString(s string) ([]byte, error) {
	// quotes and b at start are guaranteed to be there
	b := make([]byte, 0, 1)
	s = s[2 : len(s)-1]
	for len(s) != 0 {
		c := s[0]
		s = s[1:]
		if c != '\\' {
			b = append(b, c)
			continue
		}
		c = s[0]
		s = s[1:]
		switch c {
		case 'a':
			b = append(b, 0x7)
		case 'b':
			b = append(b, 0x8)
		case 'f':
			b = append(b, 0xc)
		case 'n':
			b = append(b, '\n')
		case 'r':
			b = append(b, '\r')
		case 't':
			b = append(b, '\t')
		case 'x':
			if len(s) < 2 {
				return nil, errors.New("short escape")
			}
			n, err := strconv.ParseUint(s[:2], 16, 8)
			if err != nil {
				return nil, err
			}
			b = append(b, byte(n))
			s = s[2:]
		case '0':
			if len(s) < 3 {
				return nil, errors.New("short escape")
			}
			n, err := strconv.ParseUint(s[:3], 8, 8)
			if err != nil {
				return nil, err
			}
			b = append(b, byte(n))
			s = s[3:]
		default:
			b = append(b, c)
		}
	}
	return append(b, 0), nil
}

func varInfer(n varNode) (Signature, error) {
	if sig, ok := n.Sigs().Single(); ok {
		return sig, nil
	}
	return n.Infer()
}
