# Is This Path Inside This Other Path?

It turns out this question isn't trivial to answer using Node's built-in path APIs. A naive `indexOf`-based solution will fail sometimes on Windows, which is case-insensitive (see e.g. [isaacs/npm#4214][]). You might then think to be clever with `path.resolve`, but you have to be careful to account for situations whether the paths have different drive letters, or else you'll cause bugs like [isaacs/npm#4313][]. And let's not even get started on trailing slashes.

The **path-is-inside** package will give you a robust, cross-platform way of detecting whether a given path is inside another path.

## Usage

Pretty simple. First the path being tested; then the potential parent. Like so:

```js
var pathIsInside = require("path-is-inside");

pathIsInside("/x/y/z", "/x/y") // true
pathIsInside("/x/y", "/x/y/z") // false
```

## OS-Specific Behavior

Like Node's built-in path module, path-is-inside treats all file paths on Windows as case-insensitive, whereas it treats all file paths on *-nix operating systems as case-sensitive. Keep this in mind especially when working on a Mac, where, despite Node's defaults, the OS usually treats paths case-insensitively.

In practice, this means:

```js
// On Windows

pathIsInside("C:\\X\\Y\\Z", "C:\\x\\y") // true

// On *-nix, including Mac OS X

pathIsInside("/X/Y/Z", "/x/y") // false
```

[isaacs/npm#4214]: https://github.com/isaacs/npm/pull/4214
[isaacs/npm#4313]: https://github.com/isaacs/npm/issues/4313
