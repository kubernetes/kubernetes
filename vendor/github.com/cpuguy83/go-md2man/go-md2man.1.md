go-md2man 1 "January 2015" go-md2man "User Manual"
==================================================

# NAME
  go-md2man - Convert mardown files into manpages

# SYNOPSIS
  go-md2man -in=[/path/to/md/file] -out=[/path/to/output]

# Description
  go-md2man converts standard markdown formatted documents into manpages. It is
  written purely in Go so as to reduce dependencies on 3rd party libs.

# Example
  Convert the markdown file "go-md2man.1.md" into a manpage.

    go-md2man -in=README.md -out=go-md2man.1.out

# HISTORY
  January 2015, Originally compiled by Brian Goff( cpuguy83@gmail.com )

