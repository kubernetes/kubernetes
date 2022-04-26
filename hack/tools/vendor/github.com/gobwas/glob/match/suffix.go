package match

import (
	"fmt"
	"strings"
)

type Suffix struct {
	Suffix string
}

func NewSuffix(s string) Suffix {
	return Suffix{s}
}

func (self Suffix) Len() int {
	return lenNo
}

func (self Suffix) Match(s string) bool {
	return strings.HasSuffix(s, self.Suffix)
}

func (self Suffix) Index(s string) (int, []int) {
	idx := strings.Index(s, self.Suffix)
	if idx == -1 {
		return -1, nil
	}

	return 0, []int{idx + len(self.Suffix)}
}

func (self Suffix) String() string {
	return fmt.Sprintf("<suffix:%s>", self.Suffix)
}
