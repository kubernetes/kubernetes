package link

import (
	"fmt"
	"runtime"
)

func platformPrefix(symbol string) string {

	prefix := runtime.GOARCH

	// per https://github.com/golang/go/blob/master/src/go/build/syslist.go
	switch prefix {
	case "386":
		prefix = "ia32"
	case "amd64", "amd64p32":
		prefix = "x64"
	case "arm64", "arm64be":
		prefix = "arm64"
	default:
		return symbol
	}

	return fmt.Sprintf("__%s_%s", prefix, symbol)
}
