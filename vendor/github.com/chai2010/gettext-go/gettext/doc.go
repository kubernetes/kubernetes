// Copyright 2013 <chaishushan{AT}gmail.com>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package gettext implements a basic GNU's gettext library.

Example:
	import (
		"github.com/chai2010/gettext-go/gettext"
	)

	func main() {
		gettext.SetLocale("zh_CN")
		gettext.Textdomain("hello")

		// gettext.BindTextdomain("hello", "local", nil)         // from local dir
		// gettext.BindTextdomain("hello", "local.zip", nil)     // from local zip file
		// gettext.BindTextdomain("hello", "local.zip", zipData) // from embedded zip data

		gettext.BindTextdomain("hello", "local", nil)

		// translate source text
		fmt.Println(gettext.Gettext("Hello, world!"))
		// Output: 你好, 世界!

		// translate resource
		fmt.Println(string(gettext.Getdata("poems.txt")))
		// Output: ...
	}

Translate directory struct("../examples/local.zip"):

	Root: "path" or "file.zip/zipBaseName"
	 +-default                 # local: $(LC_MESSAGES) or $(LANG) or "default"
	 |  +-LC_MESSAGES            # just for `gettext.Gettext`
	 |  |   +-hello.mo             # $(Root)/$(local)/LC_MESSAGES/$(domain).mo
	 |  |   \-hello.po             # $(Root)/$(local)/LC_MESSAGES/$(domain).mo
	 |  |
	 |  \-LC_RESOURCE            # just for `gettext.Getdata`
	 |      +-hello                # domain map a dir in resource translate
	 |         +-favicon.ico       # $(Root)/$(local)/LC_RESOURCE/$(domain)/$(filename)
	 |         \-poems.txt
	 |
	 \-zh_CN                   # simple chinese translate
	    +-LC_MESSAGES
	    |   +-hello.mo             # try "$(domain).mo" first
	    |   \-hello.po             # try "$(domain).po" second
	    |
	    \-LC_RESOURCE
	        +-hello
	           +-favicon.ico       # try "$(local)/$(domain)/file" first
	           \-poems.txt         # try "default/$(domain)/file" second

See:
	http://en.wikipedia.org/wiki/Gettext
	http://www.gnu.org/software/gettext/manual/html_node
	http://www.gnu.org/software/gettext/manual/html_node/Header-Entry.html
	http://www.gnu.org/software/gettext/manual/html_node/PO-Files.html
	http://www.gnu.org/software/gettext/manual/html_node/MO-Files.html
	http://www.poedit.net/

Please report bugs to <chaishushan{AT}gmail.com>.
Thanks!
*/
package gettext
