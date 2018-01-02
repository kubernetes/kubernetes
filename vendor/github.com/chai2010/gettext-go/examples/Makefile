# Copyright 2013 ChaiShushan <chaishushan{AT}gmail.com>. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

default:
	msgfmt -o local/default/LC_MESSAGES/hello.mo local/default/LC_MESSAGES/hello.po
	msgfmt -o local/zh_CN/LC_MESSAGES/hello.mo   local/zh_CN/LC_MESSAGES/hello.po
	msgfmt -o local/zh_TW/LC_MESSAGES/hello.mo   local/zh_TW/LC_MESSAGES/hello.po
	7z a -tzip -scsUTF-8 local.zip local
	go run hello.go

clean:
	rm local.zip
