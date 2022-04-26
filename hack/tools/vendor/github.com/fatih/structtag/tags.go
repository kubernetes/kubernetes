package structtag

import (
	"bytes"
	"errors"
	"fmt"
	"strconv"
	"strings"
)

var (
	errTagSyntax      = errors.New("bad syntax for struct tag pair")
	errTagKeySyntax   = errors.New("bad syntax for struct tag key")
	errTagValueSyntax = errors.New("bad syntax for struct tag value")

	errKeyNotSet      = errors.New("tag key does not exist")
	errTagNotExist    = errors.New("tag does not exist")
	errTagKeyMismatch = errors.New("mismatch between key and tag.key")
)

// Tags represent a set of tags from a single struct field
type Tags struct {
	tags []*Tag
}

// Tag defines a single struct's string literal tag
type Tag struct {
	// Key is the tag key, such as json, xml, etc..
	// i.e: `json:"foo,omitempty". Here key is: "json"
	Key string

	// Name is a part of the value
	// i.e: `json:"foo,omitempty". Here name is: "foo"
	Name string

	// Options is a part of the value. It contains a slice of tag options i.e:
	// `json:"foo,omitempty". Here options is: ["omitempty"]
	Options []string
}

// Parse parses a single struct field tag and returns the set of tags.
func Parse(tag string) (*Tags, error) {
	var tags []*Tag

	hasTag := tag != ""

	// NOTE(arslan) following code is from reflect and vet package with some
	// modifications to collect all necessary information and extend it with
	// usable methods
	for tag != "" {
		// Skip leading space.
		i := 0
		for i < len(tag) && tag[i] == ' ' {
			i++
		}
		tag = tag[i:]
		if tag == "" {
			break
		}

		// Scan to colon. A space, a quote or a control character is a syntax
		// error. Strictly speaking, control chars include the range [0x7f,
		// 0x9f], not just [0x00, 0x1f], but in practice, we ignore the
		// multi-byte control characters as it is simpler to inspect the tag's
		// bytes than the tag's runes.
		i = 0
		for i < len(tag) && tag[i] > ' ' && tag[i] != ':' && tag[i] != '"' && tag[i] != 0x7f {
			i++
		}

		if i == 0 {
			return nil, errTagKeySyntax
		}
		if i+1 >= len(tag) || tag[i] != ':' {
			return nil, errTagSyntax
		}
		if tag[i+1] != '"' {
			return nil, errTagValueSyntax
		}

		key := string(tag[:i])
		tag = tag[i+1:]

		// Scan quoted string to find value.
		i = 1
		for i < len(tag) && tag[i] != '"' {
			if tag[i] == '\\' {
				i++
			}
			i++
		}
		if i >= len(tag) {
			return nil, errTagValueSyntax
		}

		qvalue := string(tag[:i+1])
		tag = tag[i+1:]

		value, err := strconv.Unquote(qvalue)
		if err != nil {
			return nil, errTagValueSyntax
		}

		res := strings.Split(value, ",")
		name := res[0]
		options := res[1:]
		if len(options) == 0 {
			options = nil
		}

		tags = append(tags, &Tag{
			Key:     key,
			Name:    name,
			Options: options,
		})
	}

	if hasTag && len(tags) == 0 {
		return nil, nil
	}

	return &Tags{
		tags: tags,
	}, nil
}

// Get returns the tag associated with the given key. If the key is present
// in the tag the value (which may be empty) is returned. Otherwise the
// returned value will be the empty string. The ok return value reports whether
// the tag exists or not (which the return value is nil).
func (t *Tags) Get(key string) (*Tag, error) {
	for _, tag := range t.tags {
		if tag.Key == key {
			return tag, nil
		}
	}

	return nil, errTagNotExist
}

// Set sets the given tag. If the tag key already exists it'll override it
func (t *Tags) Set(tag *Tag) error {
	if tag.Key == "" {
		return errKeyNotSet
	}

	added := false
	for i, tg := range t.tags {
		if tg.Key == tag.Key {
			added = true
			t.tags[i] = tag
		}
	}

	if !added {
		// this means this is a new tag, add it
		t.tags = append(t.tags, tag)
	}

	return nil
}

// AddOptions adds the given option for the given key. If the option already
// exists it doesn't add it again.
func (t *Tags) AddOptions(key string, options ...string) {
	for i, tag := range t.tags {
		if tag.Key != key {
			continue
		}

		for _, opt := range options {
			if !tag.HasOption(opt) {
				tag.Options = append(tag.Options, opt)
			}
		}

		t.tags[i] = tag
	}
}

// DeleteOptions deletes the given options for the given key
func (t *Tags) DeleteOptions(key string, options ...string) {
	hasOption := func(option string) bool {
		for _, opt := range options {
			if opt == option {
				return true
			}
		}
		return false
	}

	for i, tag := range t.tags {
		if tag.Key != key {
			continue
		}

		var updated []string
		for _, opt := range tag.Options {
			if !hasOption(opt) {
				updated = append(updated, opt)
			}
		}

		tag.Options = updated
		t.tags[i] = tag
	}
}

// Delete deletes the tag for the given keys
func (t *Tags) Delete(keys ...string) {
	hasKey := func(key string) bool {
		for _, k := range keys {
			if k == key {
				return true
			}
		}
		return false
	}

	var updated []*Tag
	for _, tag := range t.tags {
		if !hasKey(tag.Key) {
			updated = append(updated, tag)
		}
	}

	t.tags = updated
}

// Tags returns a slice of tags. The order is the original tag order unless it
// was changed.
func (t *Tags) Tags() []*Tag {
	return t.tags
}

// Tags returns a slice of tags. The order is the original tag order unless it
// was changed.
func (t *Tags) Keys() []string {
	var keys []string
	for _, tag := range t.tags {
		keys = append(keys, tag.Key)
	}
	return keys
}

// String reassembles the tags into a valid literal tag field representation
func (t *Tags) String() string {
	tags := t.Tags()
	if len(tags) == 0 {
		return ""
	}

	var buf bytes.Buffer
	for i, tag := range t.Tags() {
		buf.WriteString(tag.String())
		if i != len(tags)-1 {
			buf.WriteString(" ")
		}
	}
	return buf.String()
}

// HasOption returns true if the given option is available in options
func (t *Tag) HasOption(opt string) bool {
	for _, tagOpt := range t.Options {
		if tagOpt == opt {
			return true
		}
	}

	return false
}

// Value returns the raw value of the tag, i.e. if the tag is
// `json:"foo,omitempty", the Value is "foo,omitempty"
func (t *Tag) Value() string {
	options := strings.Join(t.Options, ",")
	if options != "" {
		return fmt.Sprintf(`%s,%s`, t.Name, options)
	}
	return t.Name
}

// String reassembles the tag into a valid tag field representation
func (t *Tag) String() string {
	return fmt.Sprintf(`%s:%q`, t.Key, t.Value())
}

// GoString implements the fmt.GoStringer interface
func (t *Tag) GoString() string {
	template := `{
		Key:    '%s',
		Name:   '%s',
		Option: '%s',
	}`

	if t.Options == nil {
		return fmt.Sprintf(template, t.Key, t.Name, "nil")
	}

	options := strings.Join(t.Options, ",")
	return fmt.Sprintf(template, t.Key, t.Name, options)
}

func (t *Tags) Len() int {
	return len(t.tags)
}

func (t *Tags) Less(i int, j int) bool {
	return t.tags[i].Key < t.tags[j].Key
}

func (t *Tags) Swap(i int, j int) {
	t.tags[i], t.tags[j] = t.tags[j], t.tags[i]
}
