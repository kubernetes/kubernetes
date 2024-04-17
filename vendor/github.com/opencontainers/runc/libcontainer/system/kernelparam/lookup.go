package kernelparam

import (
	"io/fs"
	"strings"
)

func runeFilter(c rune) bool {
	return c < '!' || c > '~'
}

// LookupKernelBootParameters returns the selected kernel parameters specified
// in the kernel command line. The parameters are returned as a map of key-value pairs.
func LookupKernelBootParameters(rootFS fs.FS, lookupParameters ...string) (map[string]string, error) {
	cmdline, err := fs.ReadFile(rootFS, "proc/cmdline")
	if err != nil {
		return nil, err
	}

	kernelParameters := make(map[string]string)
	remaining := len(lookupParameters)

	for _, parameter := range strings.FieldsFunc(string(cmdline), runeFilter) {
		if remaining == 0 {
			break
		}
		idx := strings.IndexByte(parameter, '=')
		if idx == -1 {
			continue
		}
		for _, lookupParam := range lookupParameters {
			if lookupParam == parameter[:idx] {
				kernelParameters[lookupParam] = parameter[idx+1:]
				remaining--
				break
			}
		}
	}

	return kernelParameters, nil
}
