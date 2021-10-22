// +build codegen

package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/aws/aws-sdk-go/private/model/api"
)

func main() {
	glob := filepath.FromSlash(os.Args[1])
	modelPaths, err := api.ExpandModelGlobPath(glob)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to expand glob, %v\n", err)
		os.Exit(1)
	}

	_, excluded := api.TrimModelServiceVersions(modelPaths)

	for _, exclude := range excluded {
		modelPath := filepath.Dir(exclude)
		fmt.Println("removing:", modelPath)
		os.RemoveAll(modelPath)
	}
}
