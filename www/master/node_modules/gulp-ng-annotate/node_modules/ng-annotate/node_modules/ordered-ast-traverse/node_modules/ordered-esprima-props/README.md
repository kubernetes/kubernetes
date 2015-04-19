# ordered-esprima-props
A map from type (string) to an array of property names (strings)
in lexical order, i.e. an AST-traversal in this order will visit
nodes in increasing source code position.

Tested with Esprima but should work for any Mozilla Parser API
compatible AST, see
https://developer.mozilla.org/en-US/docs/Mozilla/Projects/SpiderMonkey/Parser_API

## License
`MIT`, see [LICENSE](LICENSE) file.
