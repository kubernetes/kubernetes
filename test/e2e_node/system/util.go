/*
Copyright 2016 The Kubernetes Authors.

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

package system

import (
	"fmt"
)

// resultType is type of the validation result. Different validation results
// corresponds to different colors.
type resultType int

const (
	good resultType = iota
	bad
	warn
)

// color is the color of the message.
type color int

const (
	red    color = 31
	green        = 32
	yellow       = 33
	white        = 37
)

func wrap(s string, c color) string {
	return fmt.Sprintf("\033[0;%dm%s\033[0m", c, s)
}

// report reports "item: r". item is white, and the color of r depends on the
// result type.
func report(item, r string, t resultType) {
	var c color
	switch t {
	case good:
		c = green
	case bad:
		c = red
	case warn:
		c = yellow
	default:
		c = white
	}
	fmt.Printf("%s: %s\n", wrap(item, white), wrap(r, c))
}
