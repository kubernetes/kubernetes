# gotextdiff - unified text diffing in Go <a href="https://hexops.com"><img align="right" alt="Hexops logo" src="https://raw.githubusercontent.com/hexops/media/master/readme.svg"></img></a>

This is a copy of the Go text diffing packages that [the official Go language server gopls uses internally](https://github.com/golang/tools/tree/master/internal/lsp/diff) to generate unified diffs.

If you've previously tried to generate unified text diffs in Go (like the ones you see in Git and on GitHub), you may have found [github.com/sergi/go-diff](https://github.com/sergi/go-diff) which is a Go port of Neil Fraser's google-diff-match-patch code - however it [does not support unified diffs](https://github.com/sergi/go-diff/issues/57).

This is arguably one of the best (and most maintained) unified text diffing packages in Go as of at least 2020.

(All credit goes to [the Go authors](http://tip.golang.org/AUTHORS), I am merely re-publishing their work so others can use it.)

## Example usage

Import the packages:

```Go
import (
    "github.com/hexops/gotextdiff"
    "github.com/hexops/gotextdiff/myers"
)
```

Assuming you want to diff `a.txt` and `b.txt`, whose contents are stored in `aString` and `bString` then:

```Go
edits := myers.ComputeEdits(span.URIFromPath("a.txt"), aString, bString)
diff := fmt.Sprint(gotextdiff.ToUnified("a.txt", "b.txt", aString, edits))
```

`diff` will be a string like:

```diff
--- a.txt
+++ b.txt
@@ -1,13 +1,28 @@
-foo
+bar
```

## API compatability

We will publish a new major version anytime the API changes in a backwards-incompatible way. Because the upstream is not being developed with this being a public package in mind, API breakages may occur more often than in other Go packages (but you can always continue using the old version thanks to Go modules.)

## Alternatives

- [github.com/andreyvit/diff](https://github.com/andreyvit/diff): Quick'n'easy string diffing functions for Golang based on github.com/sergi/go-diff.
- [github.com/kylelemons/godebug/diff](https://github.com/kylelemons/godebug/tree/master/diff): implements a linewise diff algorithm ([inactive](https://github.com/kylelemons/godebug/issues/22#issuecomment-524573477)).

## Contributing

We will only accept changes made [upstream](https://github.com/golang/tools/tree/master/internal/lsp/diff), please send any contributions to the upstream instead! Compared to the upstream, only import paths will be modified (to be non-`internal` so they are importable.) The only thing we add here is this README.

## License

See https://github.com/golang/tools/blob/master/LICENSE
