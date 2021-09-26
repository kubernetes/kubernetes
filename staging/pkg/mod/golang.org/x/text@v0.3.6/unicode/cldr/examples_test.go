package cldr_test

import (
	"fmt"
	"os"
	"path/filepath"

	"golang.org/x/text/internal/gen"
	"golang.org/x/text/unicode/cldr"
)

func ExampleDecoder() {
	// Obtain the default CLDR reader (only for x/text packages).

	var d cldr.Decoder

	// Speed up decoding by setting filters for only what you need.
	d.SetDirFilter("main", "supplemental")
	d.SetSectionFilter("numbers", "plurals")

	cldr, err := d.DecodeZip(gen.OpenCLDRCoreZip())
	if err != nil {
		fmt.Println("ERROR", err)
		return
	}
	supplemental := cldr.Supplemental()

	fmt.Println(supplemental.MeasurementData.MeasurementSystem[0].Type)
	for _, lang := range cldr.Locales() {
		data := cldr.RawLDML(lang)
		fmt.Println(lang, data.Identity.Version.Number)
	}
}

func ExampleDecoder_DecodePath() {
	// This directory will exist if a go generate has been run in any of the
	// packages in x/text using the cldr package.
	path := filepath.FromSlash("../../DATA/cldr/" + cldr.Version)

	var d cldr.Decoder

	// Speed up decoding by setting filters for only what you need.
	d.SetDirFilter("main")
	d.SetSectionFilter("numbers")

	cldr, err := d.DecodePath(path)
	if err != nil {
		// handle error
		fmt.Println("ERROR", err)
		return
	}
	for _, lang := range cldr.Locales() {
		if numbers := cldr.RawLDML(lang).Numbers; numbers != nil {
			fmt.Println(lang, len(numbers.Symbols))
		}
	}
}

func ExampleDecoder_DecodeZip() {
	// This directory will exist if a go generate has been run in any of the
	// packages in x/text using the cldr package.
	path := filepath.FromSlash("../../DATA/cldr/" + cldr.Version)

	var d cldr.Decoder

	r, err := os.Open(filepath.Join(path, "core.zip"))
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	// Only loading supplemental data can be done much faster using a dir
	// filter.
	d.SetDirFilter("supplemental")
	cldr, err := d.DecodeZip(r)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println(cldr.Supplemental().MeasurementData.MeasurementSystem[0].Type)
}

func ExampleSlice() {
	var dr *cldr.CLDR // assume this is initialized

	x, _ := dr.LDML("en")
	cs := x.Collations.Collation
	// remove all but the default
	cldr.MakeSlice(&cs).Filter(func(e cldr.Elem) bool {
		return e.GetCommon().Type != x.Collations.Default()
	})
	for i, c := range cs {
		fmt.Println(i, c.Type)
	}
}
