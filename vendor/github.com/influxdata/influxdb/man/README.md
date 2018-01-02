# Building the Man Pages

The man pages are created with `asciidoc`, `docbook`, and `xmlto`.

## Debian/Ubuntu

This is the easiest since Debian and Ubuntu automatically install the
dependencies correctly.

```bash
$ sudo apt-get install -y build-essential asciidoc xmlto
```

You should then be able to run `make` and the man pages will be
produced.

## Mac OS X

Mac OS X also has the tools necessary to build the docs, but one of the
dependencies gets installed incorrectly and you need an environment
variable to run it correctly.

Use Homebrew to install the dependencies. There might be other methods
to get the dependencies, but that's left up to the reader if they want
to use a different package manager.

If you have Homebrew installed, you should already have the Xcode tools
and that should include `make`.

```bash
$ brew install asciidoc xmlto
```

Then set the following environment variable everytime you run `make`.

```bash
export XML_CATALOG_FILES=/usr/local/etc/xml/catalog
```
