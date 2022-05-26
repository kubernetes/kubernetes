package match

import (
	"fmt"
	"unicode/utf8"
)

type BTree struct {
	Value            Matcher
	Left             Matcher
	Right            Matcher
	ValueLengthRunes int
	LeftLengthRunes  int
	RightLengthRunes int
	LengthRunes      int
}

func NewBTree(Value, Left, Right Matcher) (tree BTree) {
	tree.Value = Value
	tree.Left = Left
	tree.Right = Right

	lenOk := true
	if tree.ValueLengthRunes = Value.Len(); tree.ValueLengthRunes == -1 {
		lenOk = false
	}

	if Left != nil {
		if tree.LeftLengthRunes = Left.Len(); tree.LeftLengthRunes == -1 {
			lenOk = false
		}
	}

	if Right != nil {
		if tree.RightLengthRunes = Right.Len(); tree.RightLengthRunes == -1 {
			lenOk = false
		}
	}

	if lenOk {
		tree.LengthRunes = tree.LeftLengthRunes + tree.ValueLengthRunes + tree.RightLengthRunes
	} else {
		tree.LengthRunes = -1
	}

	return tree
}

func (self BTree) Len() int {
	return self.LengthRunes
}

// todo?
func (self BTree) Index(s string) (int, []int) {
	return -1, nil
}

func (self BTree) Match(s string) bool {
	inputLen := len(s)

	// self.Length, self.RLen and self.LLen are values meaning the length of runes for each part
	// here we manipulating byte length for better optimizations
	// but these checks still works, cause minLen of 1-rune string is 1 byte.
	if self.LengthRunes != -1 && self.LengthRunes > inputLen {
		return false
	}

	// try to cut unnecessary parts
	// by knowledge of length of right and left part
	var offset, limit int
	if self.LeftLengthRunes >= 0 {
		offset = self.LeftLengthRunes
	}
	if self.RightLengthRunes >= 0 {
		limit = inputLen - self.RightLengthRunes
	} else {
		limit = inputLen
	}

	for offset < limit {
		// search for matching part in substring
		index, segments := self.Value.Index(s[offset:limit])
		if index == -1 {
			releaseSegments(segments)
			return false
		}

		l := s[:offset+index]
		var left bool
		if self.Left != nil {
			left = self.Left.Match(l)
		} else {
			left = l == ""
		}

		if left {
			for i := len(segments) - 1; i >= 0; i-- {
				length := segments[i]

				var right bool
				var r string
				// if there is no string for the right branch
				if inputLen <= offset+index+length {
					r = ""
				} else {
					r = s[offset+index+length:]
				}

				if self.Right != nil {
					right = self.Right.Match(r)
				} else {
					right = r == ""
				}

				if right {
					releaseSegments(segments)
					return true
				}
			}
		}

		_, step := utf8.DecodeRuneInString(s[offset+index:])
		offset += index + step

		releaseSegments(segments)
	}

	return false
}

func (self BTree) String() string {
	const n string = "<nil>"
	var l, r string
	if self.Left == nil {
		l = n
	} else {
		l = self.Left.String()
	}
	if self.Right == nil {
		r = n
	} else {
		r = self.Right.String()
	}

	return fmt.Sprintf("<btree:[%s<-%s->%s]>", l, self.Value, r)
}
