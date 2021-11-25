# Version v1.4.4

* Fix of checksum problem because of forced tag. No changes to the code.

# Version v1.4.1

* Incorporated PR 162 by by Mohammed902: "Simplify non-Windows build tag".

# Version v1.4.0

* Incorporated PR 9 by mschneider82: "Added seekinfo to Tail".

# Version v1.3.1

* Incorporated PR 7: "Fix deadlock when stopping on non-empty file/buffer",
fixes upstream issue 93.


# Version v1.3.0

* Incorporated changes of unmerged upstream PR 149 by mezzi: "added line num
to Line struct".

# Version v1.2.1

* Incorporated changes of unmerged upstream PR 128 by jadekler: "Compile-able
code in readme".
* Incorporated changes of unmerged upstream PR 130 by fgeller: "small change
to comment wording".
* Incorporated changes of unmerged upstream PR 133 by sm3142: "removed
spurious newlines from log messages".

# Version v1.2.0

* Incorporated changes of unmerged upstream PR 126 by Code-Hex: "Solved the
 problem for never return the last line if it's not followed by a newline".
* Incorporated changes of unmerged upstream PR 131 by StoicPerlman: "Remove
deprecated os.SEEK consts". The changes bumped the minimal supported Go
release to 1.9.

# Version v1.1.0

* migration to go modules.
* release of master branch of the dormant upstream, because it contains
fixes and improvement no present in the tagged release.

