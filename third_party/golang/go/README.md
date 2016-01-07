# Vendored copy of go1.5.1's standard library's go/... packages.

Q: Why did you do this awful vendoring?

A: We need to build under go 1.3 and go 1.4 (soon to be go 1.4 and go 1.5.1). A
version of the go/types package existed for go 1.4, but it does not have the
same interface as the go 1.5 package, and @lavalamp had much better luck with
the 1.5.1 package anyway.

We will get rid of this as soon as there's a more standard way to do this, or
when we roll over to supporting go 1.5 and go 1.6.

Note that the packages here were not very happy about being transplated like
this and if you do a diff you will see the changes made to get everything to
compile.
