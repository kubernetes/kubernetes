/*
Copyright 2014 The Camlistore Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package legalprint provides a printing helper for the legal package.
package legalprint

import (
	"flag"
	"fmt"
	"io"

	"camlistore.org/pkg/legal"
)

var (
	flagLegal = flag.Bool("legal", false, "show licenses")
)

// MaybePrint will print the licenses if flagLegal has been set.
// It will return the value of the flagLegal.
func MaybePrint(out io.Writer) bool {
	if !*flagLegal {
		return false
	}
	for _, text := range legal.Licenses() {
		fmt.Fprintln(out, text)
	}
	return true
}
