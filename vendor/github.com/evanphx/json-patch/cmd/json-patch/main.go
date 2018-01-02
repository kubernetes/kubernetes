package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"

	jsonpatch "github.com/evanphx/json-patch"
	flags "github.com/jessevdk/go-flags"
)

type opts struct {
	PatchFilePaths []FileFlag `long:"patch-file" short:"p" value-name:"PATH" description:"Path to file with one or more operations"`
}

func main() {
	var o opts
	_, err := flags.Parse(&o)
	if err != nil {
		log.Fatalf("error: %s\n", err)
	}

	patches := make([]jsonpatch.Patch, len(o.PatchFilePaths))

	for i, patchFilePath := range o.PatchFilePaths {
		var bs []byte
		bs, err = ioutil.ReadFile(patchFilePath.Path())
		if err != nil {
			log.Fatalf("error reading patch file: %s", err)
		}

		var patch jsonpatch.Patch
		patch, err = jsonpatch.DecodePatch(bs)
		if err != nil {
			log.Fatalf("error decoding patch file: %s", err)
		}

		patches[i] = patch
	}

	doc, err := ioutil.ReadAll(os.Stdin)
	if err != nil {
		log.Fatalf("error reading from stdin: %s", err)
	}

	mdoc := doc
	for _, patch := range patches {
		mdoc, err = patch.Apply(mdoc)
		if err != nil {
			log.Fatalf("error applying patch: %s", err)
		}
	}

	fmt.Printf("%s", mdoc)
}
