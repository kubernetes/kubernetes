[![Build Status](https://travis-ci.org/client9/misspell.svg?branch=master)](https://travis-ci.org/client9/misspell) [![Go Report Card](https://goreportcard.com/badge/github.com/client9/misspell)](https://goreportcard.com/report/github.com/client9/misspell) [![GoDoc](https://godoc.org/github.com/client9/misspell?status.svg)](https://godoc.org/github.com/client9/misspell) [![Coverage](http://gocover.io/_badge/github.com/client9/misspell)](http://gocover.io/github.com/client9/misspell) [![license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://raw.githubusercontent.com/client9/misspell/master/LICENSE)

Correct commonly misspelled English words... quickly.

### Install


If you just want a binary and to start using `misspell`:

```
curl -L -o ./install-misspell.sh https://git.io/misspell
sh ./install-misspell.sh
```


Both will install as `./bin/misspell`.  You can adjust the download location using the `-b` flag.   File a ticket if you want another platform supported.


If you use [Go](https://golang.org/), the best way to run `misspell` is by using [gometalinter](#gometalinter).  Otherwise, install `misspell` the old-fashioned way:

```
go get -u github.com/client9/misspell/cmd/misspell
```

and misspell will be in your `GOPATH`


Also if you like to live dangerously, one could do

```bash
curl -L https://git.io/misspell | bash
```

### Usage


```bash
$ misspell all.html your.txt important.md files.go
your.txt:42:10 found "langauge" a misspelling of "language"

# ^ file, line, column
```

```
$ misspell -help
Usage of misspell:
  -debug
    	Debug matching, very slow
  -error
    	Exit with 2 if misspelling found
  -f string
    	'csv', 'sqlite3' or custom Golang template for output
  -i string
    	ignore the following corrections, comma separated
  -j int
    	Number of workers, 0 = number of CPUs
  -legal
    	Show legal information and exit
  -locale string
    	Correct spellings using locale perferances for US or UK.  Default is to use a neutral variety of English.  Setting locale to US will correct the British spelling of 'colour' to 'color'
  -o string
    	output file or [stderr|stdout|] (default "stdout")
  -q	Do not emit misspelling output
  -source string
    	Source mode: auto=guess, go=golang source, text=plain or markdown-like text (default "auto")
  -w	Overwrite file with corrections (default is just to display)
```

## FAQ

* [Automatic Corrections](#correct)
* [Converting UK spellings to US](#locale)
* [Using pipes and stdin](#stdin)
* [Golang special support](#golang)
* [gometalinter support](#gometalinter)
* [CSV Output](#csv)
* [Using SQLite3](#sqlite)
* [Changing output format](#output)
* [Checking a folder recursively](#recursive)
* [Performance](#performance)
* [Known Issues](#issues)
* [Debugging](#debug)
* [False Negatives and missing words](#missing)
* [Origin of Word Lists](#words)
* [Software License](#license)
* [Problem statement](#problem)
* [Other spelling correctors](#others)
* [Other ideas](#otherideas)

<a name="correct"></a>
### How can I make the corrections automatically?

Just add the `-w` flag!

```
$ misspell -w all.html your.txt important.md files.go
your.txt:9:21:corrected "langauge" to "language"

# ^ File is rewritten only if a misspelling is found
```

<a name="locale"></a>
### How do I convert British spellings to American (or vice-versa)?

Add the `-locale US` flag!

```bash
$ misspell -locale US important.txt
important.txt:10:20 found "colour" a misspelling of "color"
```

Add the `-locale UK` flag!

```bash
$ echo "My favorite color is blue" | misspell -locale UK
stdin:1:3:found "favorite color" a misspelling of "favourite colour"
```

Help is appreciated as I'm neither British nor an
expert in the English language.

<a name="recursive"></a>
### How do you check an entire folder recursively?

Just list a directory you'd like to check

```bash
misspell .
misspell aDirectory anotherDirectory aFile
```

You can also run misspell recursively using the following shell tricks:

```bash
misspell directory/**/*
```

or

```bash
find . -type f | xargs misspell
```

You can select a type of file as well.  The following examples selects all `.txt` files that are *not* in the `vendor` directory:

```bash
find . -type f -name '*.txt' | grep -v vendor/ | xargs misspell -error
```

<a name="stdin"></a>
### Can I use pipes or `stdin` for input?

Yes!

Print messages to `stderr` only:

```bash
$ echo "zeebra" | misspell
stdin:1:0:found "zeebra" a misspelling of "zebra"
```

Print messages to `stderr`, and corrected text to `stdout`:

```bash
$ echo "zeebra" | misspell -w
stdin:1:0:corrected "zeebra" to "zebra"
zebra
```

Only print the corrected text to `stdout`:

```bash
$ echo "zeebra" | misspell -w -q
zebra
```

<a name="golang"></a>
### Are there special rules for golang source files?

Yes!  If the file ends in `.go`, then misspell will only check spelling in
comments.

If you want to force a file to be checked as a golang source, use `-source=go`
on the command line.  Conversely, you can check a golang source as if it were
pure text by using `-source=text`.  You might want to do this since many
variable names have misspellings in them!

### Can I check only-comments in other other programming languages?

I'm told the using `-source=go` works well for ruby, javascript, java, c and
c++.

It doesn't work well for python and bash.

<a name="gometalinter"></a>
### Does this work with gometalinter?

[gometalinter](https://github.com/alecthomas/gometalinter) runs
multiple golang linters.  Starting on [2016-06-12](https://github.com/alecthomas/gometalinter/pull/134)
gometalinter supports `misspell` natively but it is disabled by default.

```bash
# update your copy of gometalinter
go get -u github.com/alecthomas/gometalinter

# install updates and misspell
gometalinter --install --update
```

To use, just enable `misspell`

```
gometalinter --enable misspell ./...
```

Note that gometalinter only checks golang files, and uses the default options
of `misspell`

You may wish to run this on your plaintext (.txt) and/or markdown files too.


<a name="csv"></a>
### How Can I Get CSV Output?

Using `-f csv`, the output is standard comma-seprated values with headers in the first row.

```
misspell -f csv *
file,line,column,typo,corrected
"README.md",9,22,langauge,language
"README.md",47,25,langauge,language
```

<a name="sqlite"></a>
### How can I export to SQLite3? 

Using `-f sqlite`, the output is a [sqlite3](https://www.sqlite.org/index.html) dump-file.

```bash
$ misspell -f sqlite * > /tmp/misspell.sql
$ cat /tmp/misspell.sql

PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE misspell(
  "file" TEXT,
  "line" INTEGER,i
  "column" INTEGER,i
  "typo" TEXT,
  "corrected" TEXT
);
INSERT INTO misspell VALUES("install.txt",202,31,"immediatly","immediately");
# etc...
COMMIT;
```

```bash
$ sqlite3 -init /tmp/misspell.sql :memory: 'select count(*) from misspell'
1
```

With some tricks you can directly pipe output to sqlite3 by using `-init /dev/stdin`:

```
misspell -f sqlite * | sqlite3 -init /dev/stdin -column -cmd '.width 60 15' ':memory' \
    'select substr(file,35),typo,count(*) as count from misspell group by file, typo order by count desc;'
```

<a name="ignore"></a>
### How can I ignore rules?

Using the `-i "comma,separated,rules"` flag you can specify corrections to ignore.

For example, if you were to run `misspell -w -error -source=text` against document that contains the string `Guy Finkelshteyn Braswell`, misspell would change the text to `Guy Finkelstheyn Bras well`.  You can then
determine the rules to ignore by reverting the change and running the with the `-debug` flag.  You can then see
that the corrections were `htey -> they` and `aswell -> as well`. To ignore these two rules, you add `-i "htey,aswell"` to
your command. With debug mode on, you can see it print the corrections, but it will no longer make them.

<a name="output"></a>
### How can I change the output format?

Using the `-f template` flag you can pass in a
[golang text template](https://golang.org/pkg/text/template/) to format the output.

One can use `printf "%q" VALUE` to safely quote a value.

The default template is compatible with [gometalinter](https://github.com/alecthomas/gometalinter)
```
{{ .Filename }}:{{ .Line }}:{{ .Column }}:corrected {{ printf "%q" .Original }} to "{{ printf "%q" .Corrected }}"
```

To just print probable misspellings:

```
-f '{{ .Original }}'
```

<a name="problem"></a>
### What problem does this solve?

This corrects commonly misspelled English words in computer source
code, and other text-based formats (`.txt`, `.md`, etc).

It is designed to run quickly so it can be
used as a [pre-commit hook](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks)
with minimal burden on the developer.

It does not work with binary formats (e.g. Word, etc).

It is not a complete spell-checking program nor a grammar checker.

<a name="others"></a>
### What are other misspelling correctors and what's wrong with them?

Some other misspelling correctors:

* https://github.com/vlajos/misspell_fixer
* https://github.com/lyda/misspell-check
* https://github.com/lucasdemarchi/codespell

They all work but had problems that prevented me from using them at scale:

* slow, all of the above check one misspelling at a time (i.e. linear) using regexps
* not MIT/Apache2 licensed (or equivalent)
* have dependencies that don't work for me (python3, bash, linux sed, etc)
* don't understand American vs. British English and sometimes makes unwelcome "corrections"

That said, they might be perfect for you and many have more features
than this project!

<a name="performance"></a>
### How fast is it?

Misspell is easily 100x to 1000x faster than other spelling correctors.  You
should be able to check and correct 1000 files in under 250ms.

This uses the mighty power of golang's
[strings.Replacer](https://golang.org/pkg/strings/#Replacer) which is
a implementation or variation of the
[Aho–Corasick algorithm](https://en.wikipedia.org/wiki/Aho–Corasick_algorithm).
This makes multiple substring matches *simultaneously*.

In addition this uses multiple CPU cores to work on multiple files.

<a name="issues"></a>
### What problems does it have?

Unlike the other projects, this doesn't know what a "word" is.  There may be
more false positives and false negatives due to this.  On the other hand, it
sometimes catches things others don't.

Either way, please file bugs and we'll fix them!

Since it operates in parallel to make corrections, it can be non-obvious to
determine exactly what word was corrected.

<a name="debug"></a>
### It's making mistakes.  How can I debug?

Run using `-debug` flag on the file you want.  It should then print what word
it is trying to correct.  Then [file a
bug](https://github.com/client9/misspell/issues) describing the problem.
Thanks!

<a name="missing"></a>
### Why is it making mistakes or missing items in golang files?

The matching function is *case-sensitive*, so variable names that are multiple
worlds either in all-upper or all-lower case sometimes can cause false
positives.  For instance a variable named `bodyreader` could trigger a false
positive since `yrea` is in the middle that could be corrected to `year`.
Other problems happen if the variable name uses a English contraction that
should use an apostrophe.  The best way of fixing this is to use the
[Effective Go naming
conventions](https://golang.org/doc/effective_go.html#mixed-caps) and use
[camelCase](https://en.wikipedia.org/wiki/CamelCase) for variable names.  You
can check your code using [golint](https://github.com/golang/lint)

<a name="license"></a>
### What license is this?

The main code is [MIT](https://github.com/client9/misspell/blob/master/LICENSE).

Misspell also makes uses of the Golang standard library and contains a modified version of Golang's [strings.Replacer](https://golang.org/pkg/strings/#Replacer)
which are covered under a [BSD License](https://github.com/golang/go/blob/master/LICENSE).  Type `misspell -legal` for more details or see [legal.go](https://github.com/client9/misspell/blob/master/legal.go)

<a name="words"></a>
### Where do the word lists come from?

It started with a word list from
[Wikipedia](https://en.wikipedia.org/wiki/Wikipedia:Lists_of_common_misspellings/For_machines).
Unfortunately, this list had to be highly edited as many of the words are
obsolete or based from mistakes on mechanical typewriters (I'm guessing).

Additional words were added based on actually mistakes seen in
the wild (meaning self-generated).

Variations of UK and US spellings are based on many sources including:

* http://www.tysto.com/uk-us-spelling-list.html (with heavy editing, many are incorrect)
* http://www.oxforddictionaries.com/us/words/american-and-british-spelling-american (excellent site but incomplete)
* Diffing US and UK [scowl dictionaries](http://wordlist.aspell.net)

American English is more accepting of spelling variations than is British
English, so "what is American or not" is subject to opinion.  Corrections and help welcome.

<a name="otherideas"></a>
### What are some other enhancements that could be done?

Here's some ideas for enhancements:

*Capitalization of proper nouns* could be done (e.g. weekday and month names, country names, language names)

*Opinionated US spellings*   US English has a number of words with alternate
spellings.  Think [adviser vs.
advisor](http://grammarist.com/spelling/adviser-advisor/).  While "advisor" is not wrong, the opinionated US
locale would correct "advisor" to "adviser".

*Versioning*  Some type of versioning is needed so reporting mistakes and errors is easier.

*Feedback*  Mistakes would be sent to some server for agregation and feedback review.

*Contractions and Apostrophes* This would optionally correct "isnt" to
"isn't", etc.
