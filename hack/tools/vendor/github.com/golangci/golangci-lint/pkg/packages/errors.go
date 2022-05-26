package packages

import (
	"fmt"
	"go/token"
	"strconv"
	"strings"

	"github.com/pkg/errors"
)

func ParseErrorPosition(pos string) (*token.Position, error) {
	// file:line(<optional>:colon)
	parts := strings.Split(pos, ":")
	if len(parts) == 1 {
		return nil, errors.New("no colons")
	}

	file := parts[0]
	line, err := strconv.Atoi(parts[1])
	if err != nil {
		return nil, fmt.Errorf("can't parse line number %q: %s", parts[1], err)
	}

	var column int
	if len(parts) == 3 { // no column
		column, err = strconv.Atoi(parts[2])
		if err != nil {
			return nil, errors.Wrapf(err, "failed to parse column from %q", parts[2])
		}
	}

	return &token.Position{
		Filename: file,
		Line:     line,
		Column:   column,
	}, nil
}
