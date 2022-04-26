package match

import (
	"fmt"
)

type Nothing struct{}

func NewNothing() Nothing {
	return Nothing{}
}

func (self Nothing) Match(s string) bool {
	return len(s) == 0
}

func (self Nothing) Index(s string) (int, []int) {
	return 0, segments0
}

func (self Nothing) Len() int {
	return lenZero
}

func (self Nothing) String() string {
	return fmt.Sprintf("<nothing>")
}
