// Copyright 2017 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package printer provides support for writing generated code.
package printer

import (
	"fmt"
)

const indentation = "  "

// Code represents a file of code to be printed.
type Code struct {
	text   string
	indent int
}

// Print adds a line of code using the current indentation. Accepts printf-style format strings and arguments.
func (c *Code) Print(args ...interface{}) {
	if len(args) > 0 {
		for i := 0; i < c.indent; i++ {
			c.text += indentation
		}
		c.text += fmt.Sprintf(args[0].(string), args[1:]...)
	}
	c.text += "\n"
}

// PrintIf adds a line of code using the current indentation if a condition is true. Accepts printf-style format strings and arguments.
func (c *Code) PrintIf(condition bool, args ...interface{}) {
	if !condition {
		return
	}
	if len(args) > 0 {
		for i := 0; i < c.indent; i++ {
			c.text += indentation
		}
		c.text += fmt.Sprintf(args[0].(string), args[1:]...)
	}
	c.text += "\n"
}

// String returns the accumulated code as a string.
func (c *Code) String() string {
	return c.text
}

// Indent adds one level of indentation.
func (c *Code) Indent() {
	c.indent++
}

// Outdent remvoes one level of indentation.
func (c *Code) Outdent() {
	c.indent--
	if c.indent < 0 {
		c.indent = 0
	}
}
