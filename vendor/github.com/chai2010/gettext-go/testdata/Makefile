# Copyright 2013 ChaiShushan <chaishushan{AT}gmail.com>. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

PO_FILES = $(wildcard *.po)
MO_FILES = $(patsubst %.po,%.mo,$(PO_FILES))

default: $(MO_FILES)

clean:
	rm *.mo

%.mo: %.po
	msgfmt -o $@ $<
