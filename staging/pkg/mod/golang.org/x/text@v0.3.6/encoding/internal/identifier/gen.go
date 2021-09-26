// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore
// +build ignore

package main

import (
	"bytes"
	"encoding/xml"
	"fmt"
	"io"
	"log"
	"strings"

	"golang.org/x/text/internal/gen"
)

type registry struct {
	XMLName  xml.Name `xml:"registry"`
	Updated  string   `xml:"updated"`
	Registry []struct {
		ID     string `xml:"id,attr"`
		Record []struct {
			Name string `xml:"name"`
			Xref []struct {
				Type string `xml:"type,attr"`
				Data string `xml:"data,attr"`
			} `xml:"xref"`
			Desc struct {
				Data string `xml:",innerxml"`
				// Any []struct {
				// 	Data string `xml:",chardata"`
				// } `xml:",any"`
				// Data string `xml:",chardata"`
			} `xml:"description,"`
			MIB   string   `xml:"value"`
			Alias []string `xml:"alias"`
			MIME  string   `xml:"preferred_alias"`
		} `xml:"record"`
	} `xml:"registry"`
}

func main() {
	r := gen.OpenIANAFile("assignments/character-sets/character-sets.xml")
	reg := &registry{}
	if err := xml.NewDecoder(r).Decode(&reg); err != nil && err != io.EOF {
		log.Fatalf("Error decoding charset registry: %v", err)
	}
	if len(reg.Registry) == 0 || reg.Registry[0].ID != "character-sets-1" {
		log.Fatalf("Unexpected ID %s", reg.Registry[0].ID)
	}

	w := &bytes.Buffer{}
	fmt.Fprintf(w, "const (\n")
	for _, rec := range reg.Registry[0].Record {
		constName := ""
		for _, a := range rec.Alias {
			if strings.HasPrefix(a, "cs") && strings.IndexByte(a, '-') == -1 {
				// Some of the constant definitions have comments in them. Strip those.
				constName = strings.Title(strings.SplitN(a[2:], "\n", 2)[0])
			}
		}
		if constName == "" {
			switch rec.MIB {
			case "2085":
				constName = "HZGB2312" // Not listed as alias for some reason.
			default:
				log.Fatalf("No cs alias defined for %s.", rec.MIB)
			}
		}
		if rec.MIME != "" {
			rec.MIME = fmt.Sprintf(" (MIME: %s)", rec.MIME)
		}
		fmt.Fprintf(w, "// %s is the MIB identifier with IANA name %s%s.\n//\n", constName, rec.Name, rec.MIME)
		if len(rec.Desc.Data) > 0 {
			fmt.Fprint(w, "// ")
			d := xml.NewDecoder(strings.NewReader(rec.Desc.Data))
			inElem := true
			attr := ""
			for {
				t, err := d.Token()
				if err != nil {
					if err != io.EOF {
						log.Fatal(err)
					}
					break
				}
				switch x := t.(type) {
				case xml.CharData:
					attr = "" // Don't need attribute info.
					a := bytes.Split([]byte(x), []byte("\n"))
					for i, b := range a {
						if b = bytes.TrimSpace(b); len(b) != 0 {
							if !inElem && i > 0 {
								fmt.Fprint(w, "\n// ")
							}
							inElem = false
							fmt.Fprintf(w, "%s ", string(b))
						}
					}
				case xml.StartElement:
					if x.Name.Local == "xref" {
						inElem = true
						use := false
						for _, a := range x.Attr {
							if a.Name.Local == "type" {
								use = use || a.Value != "person"
							}
							if a.Name.Local == "data" && use {
								// Patch up URLs to use https. From some links, the
								// https version is different from the http one.
								s := a.Value
								s = strings.Replace(s, "http://", "https://", -1)
								s = strings.Replace(s, "/unicode/", "/", -1)
								attr = s + " "
							}
						}
					}
				case xml.EndElement:
					inElem = false
					fmt.Fprint(w, attr)
				}
			}
			fmt.Fprint(w, "\n")
		}
		for _, x := range rec.Xref {
			switch x.Type {
			case "rfc":
				fmt.Fprintf(w, "// Reference: %s\n", strings.ToUpper(x.Data))
			case "uri":
				fmt.Fprintf(w, "// Reference: %s\n", x.Data)
			}
		}
		fmt.Fprintf(w, "%s MIB = %s\n", constName, rec.MIB)
		fmt.Fprintln(w)
	}
	fmt.Fprintln(w, ")")

	gen.WriteGoFile("mib.go", "identifier", w.Bytes())
}
