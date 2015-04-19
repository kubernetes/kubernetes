# Build instructions
ng-annotate is written in ES6 constlet style and uses defs.js to transpile
to ES5, via an optional build step, so that it can execute without the
`--harmony` flag passed to node.

The git repository contains the original constlet style source code as well
as the build scripts. It does not contain build artefacts (transpiled or
bundled source).

The build scripts populates the `build/es5` directory.
The NPM package contains a snapshot of the git repository at the time as
well as `build/es5`. `package.json` refers to the transpiled version in
`build/es5`, so there's no need to execute node with `--harmony` when
running a `npm -g` installed `ng-annotate` from the command line or when
doing a `require("ng-annotate")` of the same.

If you clone the git repository then don't forget to also `npm install` the
dependencies (see `package.json`).

If you want to run ng-annotate in its original form (rather than
transpiled), for instance if you're hacking on it, then just run the tool
via `ng-annotate-harmony` (not a NPM exported binary but check the package
root) or include it as a library via
`require("ng-annotate/ng-annotate-main")`. This applies to a git clone just
as well as the NPM package.

`run-tests.js` is the test runner. Run it on the original source via
`node --harmony run-tests.js` or `npm test`. The tests are run automatically
in the build scripts.

To build, `cd build` then run `./build.sh` for defs transpilation.
`./clean.sh` removes the build artefacts.

I use `prepare.sh` to prepare a release tarball for NPM publishing.

Happy hacking!
