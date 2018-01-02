// Copyright (c) 2014-2017 TSUYUSATO Kitsune
// This software is released under the MIT License.
// http://opensource.org/licenses/mit-license.php

package heredoc_test

import (
	"fmt"
)

import "github.com/MakeNowJust/heredoc"

func ExampleDoc_lipsum() {
	fmt.Print(heredoc.Doc(`
		Lorem ipsum dolor sit amet, consectetur adipisicing elit,
		sed do eiusmod tempor incididunt ut labore et dolore magna
		aliqua. Ut enim ad minim veniam, ...
	`))
	// Output:
	// Lorem ipsum dolor sit amet, consectetur adipisicing elit,
	// sed do eiusmod tempor incididunt ut labore et dolore magna
	// aliqua. Ut enim ad minim veniam, ...
	//
}

func ExampleDoc_spec() {
	// Single line string is no change.
	fmt.Println(heredoc.Doc(`It is single line.`))
	// If first line is empty, heredoc.Doc removes first line.
	fmt.Println(heredoc.Doc(`
		It is first line.
		It is second line.`))
	// If last line is empty and more little length than indents,
	// heredoc.Doc removes last line's content.
	fmt.Println(heredoc.Doc(`
		Next is last line.
	`))
	fmt.Println("Previous is last line.")
	// Output:
	// It is single line.
	// It is first line.
	// It is second line.
	// Next is last line.
	//
	// Previous is last line.
}

func ExampleDocf() {
	libName := "github.com/MakeNowJust/heredoc"
	author := "TSUYUSATO Kitsune (@MakeNowJust)"
	fmt.Printf(heredoc.Docf(`
		Library Name  : %s
		Author        : %s
		Repository URL: http://%s.git
	`, libName, author, libName))
	// Output:
	// Library Name  : github.com/MakeNowJust/heredoc
	// Author        : TSUYUSATO Kitsune (@MakeNowJust)
	// Repository URL: http://github.com/MakeNowJust/heredoc.git
}
