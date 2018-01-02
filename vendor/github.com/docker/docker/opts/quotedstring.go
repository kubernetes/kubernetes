package opts

// QuotedString is a string that may have extra quotes around the value. The
// quotes are stripped from the value.
type QuotedString struct {
	value *string
}

// Set sets a new value
func (s *QuotedString) Set(val string) error {
	*s.value = trimQuotes(val)
	return nil
}

// Type returns the type of the value
func (s *QuotedString) Type() string {
	return "string"
}

func (s *QuotedString) String() string {
	return string(*s.value)
}

func trimQuotes(value string) string {
	lastIndex := len(value) - 1
	for _, char := range []byte{'\'', '"'} {
		if value[0] == char && value[lastIndex] == char {
			return value[1:lastIndex]
		}
	}
	return value
}

// NewQuotedString returns a new quoted string option
func NewQuotedString(value *string) *QuotedString {
	return &QuotedString{value: value}
}
