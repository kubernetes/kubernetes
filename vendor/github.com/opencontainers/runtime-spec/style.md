# <a name="styleAndConventions" />Style and conventions

## <a name="styleOneSentence" />One sentence per line

To keep consistency throughout the Markdown files in the Open Container spec all files should be formatted one sentence per line.
This fixes two things: it makes diffing easier with git and it resolves fights about line wrapping length.
For example, this paragraph will span three lines in the Markdown source.

## <a name="styleHex" />Traditionally hex settings should use JSON integers, not JSON strings

For example, [`"classID": 1048577`](config-linux.md#network) instead of `"classID": "0x100001"`.
The config JSON isn't enough of a UI to be worth jumping through string <-> integer hoops to support an 0x… form ([source][integer-over-hex]).

## <a name="styleConstantNames" />Constant names should keep redundant prefixes

For example, `CAP_KILL` instead of `KILL` in [**`process.capabilities`**](config.md#process).
The redundancy reduction from removing the namespacing prefix is not useful enough to be worth trimming the upstream identifier ([source][keep-prefix]).

## <a name="styleOptionalSettings" />Optional settings should not have pointer Go types

Because in many cases the Go default for the type is a no-op in the spec (sources [here][no-pointer-for-strings], [here][no-pointer-for-slices], and [here][no-pointer-for-boolean]).
The exceptions are entries where we need to distinguish between “not set” and “set to the Go default for that type” ([source][pointer-when-updates-require-changes]), and this decision should be made on a per-setting case.

## Links

Internal links should be [relative links][markdown-relative-links] when linking to content within the repository.
Internal links should be used inline.

External links should be collected at the bottom of a markdown file and used as referenced links.
See 'Referenced Links' in this [markdown quick reference][markdown-quick-reference].
The use of referenced links in the markdown body helps to keep files clean and organized.
This also facilitates updates of external link targets on a per-file basis.

Referenced links should be kept in two alphabetically sorted sets, a general reference section followed by a man page section.
To keep Pandoc happy, duplicate naming of links within pages listed in the Makefile's `DOC_FILES` variable should be avoided by appending an `_N` to the link tagname, where `N` is some number not currently in use.
The organization and style of an existing reference section should be maintained unless it violates these style guidelines.

An exception to these rules is when a URL is needed contextually, for example when showing an explicit link to the reader.

## Examples

### <a name="styleAnchoring" />Anchoring

For any given section that provides a notable example, it is ideal to have it denoted with [markdown headers][markdown-headers].
The level of header should be such that it is a subheader of the header it is an example of.

#### Example

```markdown
## Some Topic

### Some Subheader

#### Further Subheader

##### Example

To use Further Subheader, ...

### Example

To use Some Topic, ...

```

### <a name="styleContent" />Content

Where necessary, the values in the example can be empty or unset, but accommodate with comments regarding this intention.

Where feasible, the content and values used in an example should convey the fullest use of the data structures concerned.
Most commonly onlookers will intend to copy-and-paste a "working example".
If the intention of the example is to be a fully utilized example, rather than a copy-and-paste example, perhaps add a comment as such.

```markdown
### Example
```
```json
{
    "foo": null,
    "bar": ""
}
```

**vs.**

```markdown
### Example

Following is a fully populated example (not necessarily for copy/paste use)
```
```json
{
    "foo": [
        1,
        2,
        3
    ],
    "bar": "waffles",
    "bif": {
        "baz": "potatoes"
    }
}
```

### Links

The following is an example of different types of links.
This is shown as a complete markdown file, where the referenced links are at the bottom.

```markdown
The specification repository's [glossary](glossary.md) is where readers can find definitions of commonly used terms.

Readers may click through to the [Open Containers namespace][open-containers] on [GitHub][github].

The URL for the Open Containers link above is: https://github.com/opencontainers


[github]: https://github.com
[open-containers]: https://github.com/opencontainers
```


[integer-over-hex]: https://github.com/opencontainers/runtime-spec/pull/267#r48360013
[keep-prefix]: https://github.com/opencontainers/runtime-spec/pull/159#issuecomment-138728337
[no-pointer-for-boolean]: https://github.com/opencontainers/runtime-spec/pull/290#r50296396
[no-pointer-for-slices]: https://github.com/opencontainers/runtime-spec/pull/316#r50782982
[no-pointer-for-strings]: https://github.com/opencontainers/runtime-spec/pull/653#issue-200439192
[pointer-when-updates-require-changes]: https://github.com/opencontainers/runtime-spec/pull/317#r50932706
[markdown-headers]: https://help.github.com/articles/basic-writing-and-formatting-syntax/#headings
[markdown-quick-reference]: https://en.support.wordpress.com/markdown-quick-reference
[markdown-relative-links]: https://help.github.com/articles/basic-writing-and-formatting-syntax/#relative-links
