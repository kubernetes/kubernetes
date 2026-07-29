// Copyright 2013 ChaiShushan <chaishushan{AT}gmail.com>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package po provides support for reading and writing GNU PO file.

Examples:
	import (
		"github.com/chai2010/gettext-go/po"
	)

	func main() {
		poFile, err := po.LoadFile("test.po")
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%v", poFile)
	}

The GNU PO file specification is at
http://www.gnu.org/software/gettext/manual/html_node/PO-Files.html.
*/
package po
