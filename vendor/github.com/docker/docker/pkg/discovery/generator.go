package discovery

import (
	"fmt"
	"regexp"
	"strconv"
)

// Generate takes care of IP generation
func Generate(pattern string) []string {
	re, _ := regexp.Compile(`\[(.+):(.+)\]`)
	submatch := re.FindStringSubmatch(pattern)
	if submatch == nil {
		return []string{pattern}
	}

	from, err := strconv.Atoi(submatch[1])
	if err != nil {
		return []string{pattern}
	}
	to, err := strconv.Atoi(submatch[2])
	if err != nil {
		return []string{pattern}
	}

	template := re.ReplaceAllString(pattern, "%d")

	var result []string
	for val := from; val <= to; val++ {
		entry := fmt.Sprintf(template, val)
		result = append(result, entry)
	}

	return result
}
