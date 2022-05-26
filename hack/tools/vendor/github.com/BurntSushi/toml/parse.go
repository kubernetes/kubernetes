package toml

import (
	"fmt"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/BurntSushi/toml/internal"
)

type parser struct {
	lx         *lexer
	context    Key      // Full key for the current hash in scope.
	currentKey string   // Base key name for everything except hashes.
	pos        Position // Current position in the TOML file.

	ordered   []Key                  // List of keys in the order that they appear in the TOML data.
	mapping   map[string]interface{} // Map keyname → key value.
	types     map[string]tomlType    // Map keyname → TOML type.
	implicits map[string]struct{}    // Record implicit keys (e.g. "key.group.names").
}

func parse(data string) (p *parser, err error) {
	defer func() {
		if r := recover(); r != nil {
			if pErr, ok := r.(ParseError); ok {
				pErr.input = data
				err = pErr
				return
			}
			panic(r)
		}
	}()

	// Read over BOM; do this here as the lexer calls utf8.DecodeRuneInString()
	// which mangles stuff.
	if strings.HasPrefix(data, "\xff\xfe") || strings.HasPrefix(data, "\xfe\xff") {
		data = data[2:]
	}

	// Examine first few bytes for NULL bytes; this probably means it's a UTF-16
	// file (second byte in surrogate pair being NULL). Again, do this here to
	// avoid having to deal with UTF-8/16 stuff in the lexer.
	ex := 6
	if len(data) < 6 {
		ex = len(data)
	}
	if i := strings.IndexRune(data[:ex], 0); i > -1 {
		return nil, ParseError{
			Message:  "files cannot contain NULL bytes; probably using UTF-16; TOML files must be UTF-8",
			Position: Position{Line: 1, Start: i, Len: 1},
			Line:     1,
			input:    data,
		}
	}

	p = &parser{
		mapping:   make(map[string]interface{}),
		types:     make(map[string]tomlType),
		lx:        lex(data),
		ordered:   make([]Key, 0),
		implicits: make(map[string]struct{}),
	}
	for {
		item := p.next()
		if item.typ == itemEOF {
			break
		}
		p.topLevel(item)
	}

	return p, nil
}

func (p *parser) panicItemf(it item, format string, v ...interface{}) {
	panic(ParseError{
		Message:  fmt.Sprintf(format, v...),
		Position: it.pos,
		Line:     it.pos.Len,
		LastKey:  p.current(),
	})
}

func (p *parser) panicf(format string, v ...interface{}) {
	panic(ParseError{
		Message:  fmt.Sprintf(format, v...),
		Position: p.pos,
		Line:     p.pos.Line,
		LastKey:  p.current(),
	})
}

func (p *parser) next() item {
	it := p.lx.nextItem()
	//fmt.Printf("ITEM %-18s line %-3d │ %q\n", it.typ, it.line, it.val)
	if it.typ == itemError {
		if it.err != nil {
			panic(ParseError{
				Position: it.pos,
				Line:     it.pos.Line,
				LastKey:  p.current(),
				err:      it.err,
			})
		}

		p.panicItemf(it, "%s", it.val)
	}
	return it
}

func (p *parser) nextPos() item {
	it := p.next()
	p.pos = it.pos
	return it
}

func (p *parser) bug(format string, v ...interface{}) {
	panic(fmt.Sprintf("BUG: "+format+"\n\n", v...))
}

func (p *parser) expect(typ itemType) item {
	it := p.next()
	p.assertEqual(typ, it.typ)
	return it
}

func (p *parser) assertEqual(expected, got itemType) {
	if expected != got {
		p.bug("Expected '%s' but got '%s'.", expected, got)
	}
}

func (p *parser) topLevel(item item) {
	switch item.typ {
	case itemCommentStart: // # ..
		p.expect(itemText)
	case itemTableStart: // [ .. ]
		name := p.nextPos()

		var key Key
		for ; name.typ != itemTableEnd && name.typ != itemEOF; name = p.next() {
			key = append(key, p.keyString(name))
		}
		p.assertEqual(itemTableEnd, name.typ)

		p.addContext(key, false)
		p.setType("", tomlHash)
		p.ordered = append(p.ordered, key)
	case itemArrayTableStart: // [[ .. ]]
		name := p.nextPos()

		var key Key
		for ; name.typ != itemArrayTableEnd && name.typ != itemEOF; name = p.next() {
			key = append(key, p.keyString(name))
		}
		p.assertEqual(itemArrayTableEnd, name.typ)

		p.addContext(key, true)
		p.setType("", tomlArrayHash)
		p.ordered = append(p.ordered, key)
	case itemKeyStart: // key = ..
		outerContext := p.context
		/// Read all the key parts (e.g. 'a' and 'b' in 'a.b')
		k := p.nextPos()
		var key Key
		for ; k.typ != itemKeyEnd && k.typ != itemEOF; k = p.next() {
			key = append(key, p.keyString(k))
		}
		p.assertEqual(itemKeyEnd, k.typ)

		/// The current key is the last part.
		p.currentKey = key[len(key)-1]

		/// All the other parts (if any) are the context; need to set each part
		/// as implicit.
		context := key[:len(key)-1]
		for i := range context {
			p.addImplicitContext(append(p.context, context[i:i+1]...))
		}

		/// Set value.
		val, typ := p.value(p.next(), false)
		p.set(p.currentKey, val, typ)
		p.ordered = append(p.ordered, p.context.add(p.currentKey))

		/// Remove the context we added (preserving any context from [tbl] lines).
		p.context = outerContext
		p.currentKey = ""
	default:
		p.bug("Unexpected type at top level: %s", item.typ)
	}
}

// Gets a string for a key (or part of a key in a table name).
func (p *parser) keyString(it item) string {
	switch it.typ {
	case itemText:
		return it.val
	case itemString, itemMultilineString,
		itemRawString, itemRawMultilineString:
		s, _ := p.value(it, false)
		return s.(string)
	default:
		p.bug("Unexpected key type: %s", it.typ)
	}
	panic("unreachable")
}

var datetimeRepl = strings.NewReplacer(
	"z", "Z",
	"t", "T",
	" ", "T")

// value translates an expected value from the lexer into a Go value wrapped
// as an empty interface.
func (p *parser) value(it item, parentIsArray bool) (interface{}, tomlType) {
	switch it.typ {
	case itemString:
		return p.replaceEscapes(it, it.val), p.typeOfPrimitive(it)
	case itemMultilineString:
		return p.replaceEscapes(it, stripFirstNewline(stripEscapedNewlines(it.val))), p.typeOfPrimitive(it)
	case itemRawString:
		return it.val, p.typeOfPrimitive(it)
	case itemRawMultilineString:
		return stripFirstNewline(it.val), p.typeOfPrimitive(it)
	case itemInteger:
		return p.valueInteger(it)
	case itemFloat:
		return p.valueFloat(it)
	case itemBool:
		switch it.val {
		case "true":
			return true, p.typeOfPrimitive(it)
		case "false":
			return false, p.typeOfPrimitive(it)
		default:
			p.bug("Expected boolean value, but got '%s'.", it.val)
		}
	case itemDatetime:
		return p.valueDatetime(it)
	case itemArray:
		return p.valueArray(it)
	case itemInlineTableStart:
		return p.valueInlineTable(it, parentIsArray)
	default:
		p.bug("Unexpected value type: %s", it.typ)
	}
	panic("unreachable")
}

func (p *parser) valueInteger(it item) (interface{}, tomlType) {
	if !numUnderscoresOK(it.val) {
		p.panicItemf(it, "Invalid integer %q: underscores must be surrounded by digits", it.val)
	}
	if numHasLeadingZero(it.val) {
		p.panicItemf(it, "Invalid integer %q: cannot have leading zeroes", it.val)
	}

	num, err := strconv.ParseInt(it.val, 0, 64)
	if err != nil {
		// Distinguish integer values. Normally, it'd be a bug if the lexer
		// provides an invalid integer, but it's possible that the number is
		// out of range of valid values (which the lexer cannot determine).
		// So mark the former as a bug but the latter as a legitimate user
		// error.
		if e, ok := err.(*strconv.NumError); ok && e.Err == strconv.ErrRange {
			p.panicItemf(it, "Integer '%s' is out of the range of 64-bit signed integers.", it.val)
		} else {
			p.bug("Expected integer value, but got '%s'.", it.val)
		}
	}
	return num, p.typeOfPrimitive(it)
}

func (p *parser) valueFloat(it item) (interface{}, tomlType) {
	parts := strings.FieldsFunc(it.val, func(r rune) bool {
		switch r {
		case '.', 'e', 'E':
			return true
		}
		return false
	})
	for _, part := range parts {
		if !numUnderscoresOK(part) {
			p.panicItemf(it, "Invalid float %q: underscores must be surrounded by digits", it.val)
		}
	}
	if len(parts) > 0 && numHasLeadingZero(parts[0]) {
		p.panicItemf(it, "Invalid float %q: cannot have leading zeroes", it.val)
	}
	if !numPeriodsOK(it.val) {
		// As a special case, numbers like '123.' or '1.e2',
		// which are valid as far as Go/strconv are concerned,
		// must be rejected because TOML says that a fractional
		// part consists of '.' followed by 1+ digits.
		p.panicItemf(it, "Invalid float %q: '.' must be followed by one or more digits", it.val)
	}
	val := strings.Replace(it.val, "_", "", -1)
	if val == "+nan" || val == "-nan" { // Go doesn't support this, but TOML spec does.
		val = "nan"
	}
	num, err := strconv.ParseFloat(val, 64)
	if err != nil {
		if e, ok := err.(*strconv.NumError); ok && e.Err == strconv.ErrRange {
			p.panicItemf(it, "Float '%s' is out of the range of 64-bit IEEE-754 floating-point numbers.", it.val)
		} else {
			p.panicItemf(it, "Invalid float value: %q", it.val)
		}
	}
	return num, p.typeOfPrimitive(it)
}

var dtTypes = []struct {
	fmt  string
	zone *time.Location
}{
	{time.RFC3339Nano, time.Local},
	{"2006-01-02T15:04:05.999999999", internal.LocalDatetime},
	{"2006-01-02", internal.LocalDate},
	{"15:04:05.999999999", internal.LocalTime},
}

func (p *parser) valueDatetime(it item) (interface{}, tomlType) {
	it.val = datetimeRepl.Replace(it.val)
	var (
		t   time.Time
		ok  bool
		err error
	)
	for _, dt := range dtTypes {
		t, err = time.ParseInLocation(dt.fmt, it.val, dt.zone)
		if err == nil {
			ok = true
			break
		}
	}
	if !ok {
		p.panicItemf(it, "Invalid TOML Datetime: %q.", it.val)
	}
	return t, p.typeOfPrimitive(it)
}

func (p *parser) valueArray(it item) (interface{}, tomlType) {
	p.setType(p.currentKey, tomlArray)

	// p.setType(p.currentKey, typ)
	var (
		types []tomlType

		// Initialize to a non-nil empty slice. This makes it consistent with
		// how S = [] decodes into a non-nil slice inside something like struct
		// { S []string }. See #338
		array = []interface{}{}
	)
	for it = p.next(); it.typ != itemArrayEnd; it = p.next() {
		if it.typ == itemCommentStart {
			p.expect(itemText)
			continue
		}

		val, typ := p.value(it, true)
		array = append(array, val)
		types = append(types, typ)

		// XXX: types isn't used here, we need it to record the accurate type
		// information.
		//
		// Not entirely sure how to best store this; could use "key[0]",
		// "key[1]" notation, or maybe store it on the Array type?
	}
	return array, tomlArray
}

func (p *parser) valueInlineTable(it item, parentIsArray bool) (interface{}, tomlType) {
	var (
		hash         = make(map[string]interface{})
		outerContext = p.context
		outerKey     = p.currentKey
	)

	p.context = append(p.context, p.currentKey)
	prevContext := p.context
	p.currentKey = ""

	p.addImplicit(p.context)
	p.addContext(p.context, parentIsArray)

	/// Loop over all table key/value pairs.
	for it := p.next(); it.typ != itemInlineTableEnd; it = p.next() {
		if it.typ == itemCommentStart {
			p.expect(itemText)
			continue
		}

		/// Read all key parts.
		k := p.nextPos()
		var key Key
		for ; k.typ != itemKeyEnd && k.typ != itemEOF; k = p.next() {
			key = append(key, p.keyString(k))
		}
		p.assertEqual(itemKeyEnd, k.typ)

		/// The current key is the last part.
		p.currentKey = key[len(key)-1]

		/// All the other parts (if any) are the context; need to set each part
		/// as implicit.
		context := key[:len(key)-1]
		for i := range context {
			p.addImplicitContext(append(p.context, context[i:i+1]...))
		}

		/// Set the value.
		val, typ := p.value(p.next(), false)
		p.set(p.currentKey, val, typ)
		p.ordered = append(p.ordered, p.context.add(p.currentKey))
		hash[p.currentKey] = val

		/// Restore context.
		p.context = prevContext
	}
	p.context = outerContext
	p.currentKey = outerKey
	return hash, tomlHash
}

// numHasLeadingZero checks if this number has leading zeroes, allowing for '0',
// +/- signs, and base prefixes.
func numHasLeadingZero(s string) bool {
	if len(s) > 1 && s[0] == '0' && !(s[1] == 'b' || s[1] == 'o' || s[1] == 'x') { // Allow 0b, 0o, 0x
		return true
	}
	if len(s) > 2 && (s[0] == '-' || s[0] == '+') && s[1] == '0' {
		return true
	}
	return false
}

// numUnderscoresOK checks whether each underscore in s is surrounded by
// characters that are not underscores.
func numUnderscoresOK(s string) bool {
	switch s {
	case "nan", "+nan", "-nan", "inf", "-inf", "+inf":
		return true
	}
	accept := false
	for _, r := range s {
		if r == '_' {
			if !accept {
				return false
			}
		}

		// isHexadecimal is a superset of all the permissable characters
		// surrounding an underscore.
		accept = isHexadecimal(r)
	}
	return accept
}

// numPeriodsOK checks whether every period in s is followed by a digit.
func numPeriodsOK(s string) bool {
	period := false
	for _, r := range s {
		if period && !isDigit(r) {
			return false
		}
		period = r == '.'
	}
	return !period
}

// Set the current context of the parser, where the context is either a hash or
// an array of hashes, depending on the value of the `array` parameter.
//
// Establishing the context also makes sure that the key isn't a duplicate, and
// will create implicit hashes automatically.
func (p *parser) addContext(key Key, array bool) {
	var ok bool

	// Always start at the top level and drill down for our context.
	hashContext := p.mapping
	keyContext := make(Key, 0)

	// We only need implicit hashes for key[0:-1]
	for _, k := range key[0 : len(key)-1] {
		_, ok = hashContext[k]
		keyContext = append(keyContext, k)

		// No key? Make an implicit hash and move on.
		if !ok {
			p.addImplicit(keyContext)
			hashContext[k] = make(map[string]interface{})
		}

		// If the hash context is actually an array of tables, then set
		// the hash context to the last element in that array.
		//
		// Otherwise, it better be a table, since this MUST be a key group (by
		// virtue of it not being the last element in a key).
		switch t := hashContext[k].(type) {
		case []map[string]interface{}:
			hashContext = t[len(t)-1]
		case map[string]interface{}:
			hashContext = t
		default:
			p.panicf("Key '%s' was already created as a hash.", keyContext)
		}
	}

	p.context = keyContext
	if array {
		// If this is the first element for this array, then allocate a new
		// list of tables for it.
		k := key[len(key)-1]
		if _, ok := hashContext[k]; !ok {
			hashContext[k] = make([]map[string]interface{}, 0, 4)
		}

		// Add a new table. But make sure the key hasn't already been used
		// for something else.
		if hash, ok := hashContext[k].([]map[string]interface{}); ok {
			hashContext[k] = append(hash, make(map[string]interface{}))
		} else {
			p.panicf("Key '%s' was already created and cannot be used as an array.", key)
		}
	} else {
		p.setValue(key[len(key)-1], make(map[string]interface{}))
	}
	p.context = append(p.context, key[len(key)-1])
}

// set calls setValue and setType.
func (p *parser) set(key string, val interface{}, typ tomlType) {
	p.setValue(key, val)
	p.setType(key, typ)
}

// setValue sets the given key to the given value in the current context.
// It will make sure that the key hasn't already been defined, account for
// implicit key groups.
func (p *parser) setValue(key string, value interface{}) {
	var (
		tmpHash    interface{}
		ok         bool
		hash       = p.mapping
		keyContext Key
	)
	for _, k := range p.context {
		keyContext = append(keyContext, k)
		if tmpHash, ok = hash[k]; !ok {
			p.bug("Context for key '%s' has not been established.", keyContext)
		}
		switch t := tmpHash.(type) {
		case []map[string]interface{}:
			// The context is a table of hashes. Pick the most recent table
			// defined as the current hash.
			hash = t[len(t)-1]
		case map[string]interface{}:
			hash = t
		default:
			p.panicf("Key '%s' has already been defined.", keyContext)
		}
	}
	keyContext = append(keyContext, key)

	if _, ok := hash[key]; ok {
		// Normally redefining keys isn't allowed, but the key could have been
		// defined implicitly and it's allowed to be redefined concretely. (See
		// the `valid/implicit-and-explicit-after.toml` in toml-test)
		//
		// But we have to make sure to stop marking it as an implicit. (So that
		// another redefinition provokes an error.)
		//
		// Note that since it has already been defined (as a hash), we don't
		// want to overwrite it. So our business is done.
		if p.isArray(keyContext) {
			p.removeImplicit(keyContext)
			hash[key] = value
			return
		}
		if p.isImplicit(keyContext) {
			p.removeImplicit(keyContext)
			return
		}

		// Otherwise, we have a concrete key trying to override a previous
		// key, which is *always* wrong.
		p.panicf("Key '%s' has already been defined.", keyContext)
	}

	hash[key] = value
}

// setType sets the type of a particular value at a given key. It should be
// called immediately AFTER setValue.
//
// Note that if `key` is empty, then the type given will be applied to the
// current context (which is either a table or an array of tables).
func (p *parser) setType(key string, typ tomlType) {
	keyContext := make(Key, 0, len(p.context)+1)
	keyContext = append(keyContext, p.context...)
	if len(key) > 0 { // allow type setting for hashes
		keyContext = append(keyContext, key)
	}
	// Special case to make empty keys ("" = 1) work.
	// Without it it will set "" rather than `""`.
	// TODO: why is this needed? And why is this only needed here?
	if len(keyContext) == 0 {
		keyContext = Key{""}
	}
	p.types[keyContext.String()] = typ
}

// Implicit keys need to be created when tables are implied in "a.b.c.d = 1" and
// "[a.b.c]" (the "a", "b", and "c" hashes are never created explicitly).
func (p *parser) addImplicit(key Key)     { p.implicits[key.String()] = struct{}{} }
func (p *parser) removeImplicit(key Key)  { delete(p.implicits, key.String()) }
func (p *parser) isImplicit(key Key) bool { _, ok := p.implicits[key.String()]; return ok }
func (p *parser) isArray(key Key) bool    { return p.types[key.String()] == tomlArray }
func (p *parser) addImplicitContext(key Key) {
	p.addImplicit(key)
	p.addContext(key, false)
}

// current returns the full key name of the current context.
func (p *parser) current() string {
	if len(p.currentKey) == 0 {
		return p.context.String()
	}
	if len(p.context) == 0 {
		return p.currentKey
	}
	return fmt.Sprintf("%s.%s", p.context, p.currentKey)
}

func stripFirstNewline(s string) string {
	if len(s) > 0 && s[0] == '\n' {
		return s[1:]
	}
	if len(s) > 1 && s[0] == '\r' && s[1] == '\n' {
		return s[2:]
	}
	return s
}

// Remove newlines inside triple-quoted strings if a line ends with "\".
func stripEscapedNewlines(s string) string {
	split := strings.Split(s, "\n")
	if len(split) < 1 {
		return s
	}

	escNL := false // Keep track of the last non-blank line was escaped.
	for i, line := range split {
		line = strings.TrimRight(line, " \t\r")

		if len(line) == 0 || line[len(line)-1] != '\\' {
			split[i] = strings.TrimRight(split[i], "\r")
			if !escNL && i != len(split)-1 {
				split[i] += "\n"
			}
			continue
		}

		escBS := true
		for j := len(line) - 1; j >= 0 && line[j] == '\\'; j-- {
			escBS = !escBS
		}
		if escNL {
			line = strings.TrimLeft(line, " \t\r")
		}
		escNL = !escBS

		if escBS {
			split[i] += "\n"
			continue
		}

		split[i] = line[:len(line)-1] // Remove \
		if len(split)-1 > i {
			split[i+1] = strings.TrimLeft(split[i+1], " \t\r")
		}
	}
	return strings.Join(split, "")
}

func (p *parser) replaceEscapes(it item, str string) string {
	replaced := make([]rune, 0, len(str))
	s := []byte(str)
	r := 0
	for r < len(s) {
		if s[r] != '\\' {
			c, size := utf8.DecodeRune(s[r:])
			r += size
			replaced = append(replaced, c)
			continue
		}
		r += 1
		if r >= len(s) {
			p.bug("Escape sequence at end of string.")
			return ""
		}
		switch s[r] {
		default:
			p.bug("Expected valid escape code after \\, but got %q.", s[r])
			return ""
		case ' ', '\t':
			p.panicItemf(it, "invalid escape: '\\%c'", s[r])
			return ""
		case 'b':
			replaced = append(replaced, rune(0x0008))
			r += 1
		case 't':
			replaced = append(replaced, rune(0x0009))
			r += 1
		case 'n':
			replaced = append(replaced, rune(0x000A))
			r += 1
		case 'f':
			replaced = append(replaced, rune(0x000C))
			r += 1
		case 'r':
			replaced = append(replaced, rune(0x000D))
			r += 1
		case '"':
			replaced = append(replaced, rune(0x0022))
			r += 1
		case '\\':
			replaced = append(replaced, rune(0x005C))
			r += 1
		case 'u':
			// At this point, we know we have a Unicode escape of the form
			// `uXXXX` at [r, r+5). (Because the lexer guarantees this
			// for us.)
			escaped := p.asciiEscapeToUnicode(it, s[r+1:r+5])
			replaced = append(replaced, escaped)
			r += 5
		case 'U':
			// At this point, we know we have a Unicode escape of the form
			// `uXXXX` at [r, r+9). (Because the lexer guarantees this
			// for us.)
			escaped := p.asciiEscapeToUnicode(it, s[r+1:r+9])
			replaced = append(replaced, escaped)
			r += 9
		}
	}
	return string(replaced)
}

func (p *parser) asciiEscapeToUnicode(it item, bs []byte) rune {
	s := string(bs)
	hex, err := strconv.ParseUint(strings.ToLower(s), 16, 32)
	if err != nil {
		p.bug("Could not parse '%s' as a hexadecimal number, but the lexer claims it's OK: %s", s, err)
	}
	if !utf8.ValidRune(rune(hex)) {
		p.panicItemf(it, "Escaped character '\\u%s' is not valid UTF-8.", s)
	}
	return rune(hex)
}
