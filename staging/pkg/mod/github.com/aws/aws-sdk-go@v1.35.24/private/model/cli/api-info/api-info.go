// +build codegen

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/aws/aws-sdk-go/private/model/api"
)

func main() {
	dir, _ := os.Open(filepath.Join("models", "apis"))
	names, _ := dir.Readdirnames(0)
	for _, name := range names {
		m, _ := filepath.Glob(filepath.Join("models", "apis", name, "*", "api-2.json"))
		if len(m) == 0 {
			continue
		}

		sort.Strings(m)
		f := m[len(m)-1]
		a := api.API{}
		a.Attach(f)
		fmt.Printf("%s\t%s\n", a.Metadata.ServiceFullName, a.Metadata.APIVersion)
	}
}
