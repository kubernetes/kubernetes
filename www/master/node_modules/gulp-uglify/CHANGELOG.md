# gulp-uglify changelog

## 1.2.0

- Update dependencies, including UglifyJS to 2.4.19.

## 1.1.0

- Fix sources path in source maps (thanks @floridoo)
- Update UglifyJS to 2.4.16 (thanks @tschaub)

## 1.0.0

- Handle cases where UglifyJS uses e.msg instead of e.message for error codes. Fixes #51.
- Supplement UglifyJS’s source map merging with vinyl-sourcemap-apply to correct issues where `sources` and `sourcesContent` were different. Fixes #43.
- Refactor option parsing and defaults, and calls to uglify-js, to reduce complexity of the main function.
- Added tests for the previously forgotten `preserveComments` option.
- Updated UglifyJS to 2.4.15.
- Changed dependencies to explicit ranges to avoid `node-semver` issues.

## 0.3.2

- Removed the PluginError factory wrapper
- Removed test that was failing due to gulp-util issue.
- Tests should end the streams they are writing to.
- Update dependencies. Fixes #44. Fixes #42.

## 0.3.1

- Fixed homepage URL in npm metadata
- Removes UglifyJS-inserted sourceMappingURL comment [Fixes #39]
- Don’t pass input source map to UglifyJS if there are no mappings
- Added installation instructions

## 0.3.0

- Removed support for old style source maps
- Added support for gulp-sourcemap
- Updated tape development dependency
- Dropped support for Node 0.9
- UglifyJS errors are no longer swallowed

## 0.2.1

- Correct source map output
- Remove `gulp` dependency by using `vinyl` in testing
- Passthrough null files correctly
- Report error if attempting to use a stream-backed file

## 0.2.0

- Dropped support for Node versions less than 0.9
- Switched to using Streams2
- Add support for generating source maps
- Add option for preserving comments
