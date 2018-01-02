# go-kmsg-parser

[![Build Status](https://travis-ci.org/euank/go-kmsg-parser.svg?branch=master)](https://travis-ci.org/euank/go-kmsg-parser)

This repository contains a library to allow parsing the `/dev/kmsg` device in
Linux. This device provides a read-write interface to the Linux Kernel's ring
buffer.

In addition to the library, a simple cli-tool that functions similarly to
`dmesg --ctime --follow` is included. This code serves both as a usage example
and as a simple way to verify it works how you'd expect on a given system.

# Contributions

Welcome

# License

Apache 2.0
