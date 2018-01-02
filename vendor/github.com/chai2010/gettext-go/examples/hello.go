// Copyright 2013 <chaishushan{AT}gmail.com>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This is a gettext-go exmaple.
package main

import (
	"fmt"

	"github.com/chai2010/gettext-go/examples/hi"
	"github.com/chai2010/gettext-go/gettext"
)

func init() {
	// bind app domain
	gettext.BindTextdomain("hello", "local", nil)
	gettext.Textdomain("hello")

	// $(LC_MESSAGES) or $(LANG) or empty
	fmt.Println(gettext.Gettext("Gettext in init."))
	fmt.Println(gettext.PGettext("main.init", "Gettext in init."))
	hi.SayHi()
	// Output(depends on local environment):
	// ?
	// ?
	// ?
	// ?

	// set simple chinese
	gettext.SetLocale("zh_CN")

	// simple chinese
	fmt.Println(gettext.Gettext("Gettext in init."))
	fmt.Println(gettext.PGettext("main.init", "Gettext in init."))
	hi.SayHi()
	// Output:
	// Init函数中的Gettext.
	// Init函数中的Gettext.
	// 来自"Hi"包的问候: 你好, 世界!
	// 来自"Hi"包的问候: 你好, 世界!
}

func main() {
	// simple chinese
	fmt.Println(gettext.Gettext("Hello, world!"))
	fmt.Println(gettext.PGettext("main.main", "Hello, world!"))
	hi.SayHi()
	// Output:
	// 你好, 世界!
	// 你好, 世界!
	// 来自"Hi"包的问候: 你好, 世界!
	// 来自"Hi"包的问候: 你好, 世界!

	// set traditional chinese
	gettext.SetLocale("zh_TW")

	// traditional chinese
	func() {
		fmt.Println(gettext.Gettext("Gettext in func."))
		fmt.Println(gettext.PGettext("main.func", "Gettext in func."))
		hi.SayHi()
		// Output:
		// 閉包函數中的Gettext.
		// 閉包函數中的Gettext.
		// 來自"Hi"包的問候: 你好, 世界!
		// 來自"Hi"包的問候: 你好, 世界!
	}()

	fmt.Println()

	// translate resource
	gettext.SetLocale("zh_CN")
	fmt.Println("poems(simple chinese):")
	fmt.Println(string(gettext.Getdata("poems.txt")))
	gettext.SetLocale("zh_TW")
	fmt.Println("poems(traditional chinese):")
	fmt.Println(string(gettext.Getdata("poems.txt")))
	gettext.SetLocale("??")
	fmt.Println("poems(default is english):")
	fmt.Println(string(gettext.Getdata("poems.txt")))
	// Output: ...
}
