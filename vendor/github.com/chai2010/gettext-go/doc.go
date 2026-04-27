// Copyright 2013 <chaishushan{AT}gmail.com>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package gettext implements a basic GNU's gettext library.

Example:
	import (
		"github.com/chai2010/gettext-go"
	)

	func main() {
		gettext.SetLanguage("zh_CN")

		// gettext.BindLocale(gettext.New("hello", "locale"))              // from locale dir
		// gettext.BindLocale(gettext.New("hello", "locale.zip"))          // from locale zip file
		// gettext.BindLocale(gettext.New("hello", "locale.zip", zipData)) // from embedded zip data

		gettext.BindLocale(gettext.New("hello", "locale"))

		// translate source text
		fmt.Println(gettext.Gettext("Hello, world!"))
		// Output: 你好, 世界!

		// translate resource
		fmt.Println(string(gettext.Getdata("poems.txt")))
		// Output: ...
	}

Translate directory struct("./examples/locale.zip"):

	Root: "path" or "file.zip/zipBaseName"
	 +-default                 # locale: $(LC_MESSAGES) or $(LANG) or "default"
	 |  +-LC_MESSAGES            # just for `gettext.Gettext`
	 |  |   +-hello.mo             # $(Root)/$(lang)/LC_MESSAGES/$(domain).mo
	 |  |   +-hello.po             # $(Root)/$(lang)/LC_MESSAGES/$(domain).po
	 |  |   \-hello.json           # $(Root)/$(lang)/LC_MESSAGES/$(domain).json
	 |  |
	 |  \-LC_RESOURCE            # just for `gettext.Getdata`
	 |      +-hello                # domain map a dir in resource translate
	 |         +-favicon.ico       # $(Root)/$(lang)/LC_RESOURCE/$(domain)/$(filename)
	 |         \-poems.txt
	 |
	 \-zh_CN                   # simple chinese translate
	    +-LC_MESSAGES
	    |   +-hello.po             # try "$(domain).po" first
	    |   +-hello.mo             # try "$(domain).mo" second
	    |   \-hello.json           # try "$(domain).json" third
	    |
	    \-LC_RESOURCE
	        +-hello
	           +-favicon.ico       # $(lang)/$(domain)/favicon.ico
	           \-poems.txt         # $(lang)/$(domain)/poems.txt

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
