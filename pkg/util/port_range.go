package util

import (
	"fmt"
	"strconv"
	"strings"
)

type PortRange struct {
	MinInclusive int
	MaxExclusive int
}

func (pr *PortRange) Size() int {
	return pr.MaxExclusive - pr.MinInclusive
}

func (pr *PortRange) Contains(p int) bool {
	return (p >= pr.MinInclusive) && (p < pr.MaxExclusive)
}

func (pr PortRange) String() string {
	// I'm guessing [a,b) is not user-friendly...
	if pr.MinInclusive == pr.MaxExclusive {
		return ""
	}
	return fmt.Sprintf("%d-%d", pr.MinInclusive, pr.MaxExclusive-1)
}

func (pr *PortRange) Set(value string) error {
	if value == "" {
		pr.MinInclusive = 0
		pr.MaxExclusive = 0
		return nil
	}

	hyphenIndex := strings.Index(value, "-")
	if hyphenIndex == -1 {
		return fmt.Errorf("Expected hyphen (-) in port range")
	}

	var err error
	pr.MinInclusive, err = strconv.Atoi(value[:hyphenIndex])
	if err == nil {
		pr.MaxExclusive, err = strconv.Atoi(value[hyphenIndex+1:])
	}
	if err != nil {
		return fmt.Errorf("Unable to parse port range: %s", value)
	}

	// Humans specify inclusive ranges
	pr.MaxExclusive++

	return nil
}

func (*PortRange) Type() string {
	return "portRange"
}
