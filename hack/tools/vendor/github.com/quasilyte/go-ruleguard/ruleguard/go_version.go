package ruleguard

import (
	"fmt"
	"go/token"
	"strconv"
	"strings"
)

type GoVersion struct {
	Major int
	Minor int
}

func (ver GoVersion) IsAny() bool { return ver.Major == 0 }

func ParseGoVersion(version string) (GoVersion, error) {
	var result GoVersion
	if version == "" {
		return GoVersion{}, nil
	}
	parts := strings.Split(version, ".")
	if len(parts) != 2 {
		return result, fmt.Errorf("invalid format: %s", version)
	}
	major, err := strconv.Atoi(parts[0])
	if err != nil {
		return result, fmt.Errorf("invalid major version part: %s: %s", parts[0], err)
	}
	minor, err := strconv.Atoi(parts[1])
	if err != nil {
		return result, fmt.Errorf("invalid minor version part: %s: %s", parts[1], err)
	}
	result.Major = major
	result.Minor = minor
	return result, nil
}

func versionCompare(x GoVersion, op token.Token, y GoVersion) bool {
	switch op {
	case token.EQL: // ==
		return x.Major == y.Major && x.Minor == y.Minor
	case token.NEQ: // !=
		return !versionCompare(x, token.EQL, y)

	case token.GTR: // >
		return x.Major > y.Major || (x.Major == y.Major && x.Minor > y.Minor)
	case token.GEQ: // >=
		return x.Major > y.Major || (x.Major == y.Major && x.Minor >= y.Minor)
	case token.LSS: // <
		return x.Major < y.Major || (x.Major == y.Major && x.Minor < y.Minor)
	case token.LEQ: // <=
		return x.Major < y.Major || (x.Major == y.Major && x.Minor <= y.Minor)

	default:
		panic("unexpected version compare op")
	}
}
