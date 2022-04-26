package match

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

// raw represents raw string to match
type Text struct {
	Str         string
	RunesLength int
	BytesLength int
	Segments    []int
}

func NewText(s string) Text {
	return Text{
		Str:         s,
		RunesLength: utf8.RuneCountInString(s),
		BytesLength: len(s),
		Segments:    []int{len(s)},
	}
}

func (self Text) Match(s string) bool {
	return self.Str == s
}

func (self Text) Len() int {
	return self.RunesLength
}

func (self Text) Index(s string) (int, []int) {
	index := strings.Index(s, self.Str)
	if index == -1 {
		return -1, nil
	}

	return index, self.Segments
}

func (self Text) String() string {
	return fmt.Sprintf("<text:`%v`>", self.Str)
}
