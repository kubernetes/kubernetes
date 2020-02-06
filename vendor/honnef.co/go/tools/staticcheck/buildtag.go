package staticcheck

import (
	"go/ast"
	"strings"

	. "honnef.co/go/tools/lint/lintdsl"
)

func buildTags(f *ast.File) [][]string {
	var out [][]string
	for _, line := range strings.Split(Preamble(f), "\n") {
		if !strings.HasPrefix(line, "+build ") {
			continue
		}
		line = strings.TrimSpace(strings.TrimPrefix(line, "+build "))
		fields := strings.Fields(line)
		out = append(out, fields)
	}
	return out
}
