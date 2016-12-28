// Copyright 2013 ChaiShushan <chaishushan{AT}gmail.com>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package mo provides support for reading and writing GNU MO file.

Examples:
	import (
		"github.com/chai2010/gettext-go/gettext/mo"
	)

	func main() {
		moFile, err := mo.Load("test.mo")
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%v", moFile)
	}

GNU MO file struct:

	        byte
	             +------------------------------------------+
	          0  | magic number = 0x950412de                |
	             |                                          |
	          4  | file format revision = 0                 |
	             |                                          |
	          8  | number of strings                        |  == N
	             |                                          |
	         12  | offset of table with original strings    |  == O
	             |                                          |
	         16  | offset of table with translation strings |  == T
	             |                                          |
	         20  | size of hashing table                    |  == S
	             |                                          |
	         24  | offset of hashing table                  |  == H
	             |                                          |
	             .                                          .
	             .    (possibly more entries later)         .
	             .                                          .
	             |                                          |
	          O  | length & offset 0th string  ----------------.
	      O + 8  | length & offset 1st string  ------------------.
	              ...                                    ...   | |
	O + ((N-1)*8)| length & offset (N-1)th string           |  | |
	             |                                          |  | |
	          T  | length & offset 0th translation  ---------------.
	      T + 8  | length & offset 1st translation  -----------------.
	              ...                                    ...   | | | |
	T + ((N-1)*8)| length & offset (N-1)th translation      |  | | | |
	             |                                          |  | | | |
	          H  | start hash table                         |  | | | |
	              ...                                    ...   | | | |
	  H + S * 4  | end hash table                           |  | | | |
	             |                                          |  | | | |
	             | NUL terminated 0th string  <----------------' | | |
	             |                                          |    | | |
	             | NUL terminated 1st string  <------------------' | |
	             |                                          |      | |
	              ...                                    ...       | |
	             |                                          |      | |
	             | NUL terminated 0th translation  <---------------' |
	             |                                          |        |
	             | NUL terminated 1st translation  <-----------------'
	             |                                          |
	              ...                                    ...
	             |                                          |
	             +------------------------------------------+

The GNU MO file specification is at
http://www.gnu.org/software/gettext/manual/html_node/MO-Files.html.
*/
package mo
