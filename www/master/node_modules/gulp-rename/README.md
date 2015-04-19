# gulp-rename

gulp-rename is a [gulp](https://github.com/wearefractal/gulp) plugin to rename files easily.

[![NPM](https://nodei.co/npm/gulp-rename.png?downloads=true&downloadRank=true&stars=true)](https://nodei.co/npm/gulp-rename/)

[![build status](https://secure.travis-ci.org/hparra/gulp-rename.svg)](http://travis-ci.org/hparra/gulp-rename)
[![devDependency Status](https://david-dm.org/hparra/gulp-rename/dev-status.svg)](https://david-dm.org/hparra/gulp-rename#info=devDependencies)

## Usage

gulp-rename provides simple file renaming methods.

```javascript
var rename = require("gulp-rename");

// rename via string
gulp.src("./src/main/text/hello.txt")
  .pipe(rename("main/text/ciao/goodbye.md"))
  .pipe(gulp.dest("./dist")); // ./dist/main/text/ciao/goodbye.md

// rename via function
gulp.src("./src/**/hello.txt")
  .pipe(rename(function (path) {
    path.dirname += "/ciao";
    path.basename += "-goodbye";
    path.extname = ".md"
  }))
  .pipe(gulp.dest("./dist")); // ./dist/main/text/ciao/hello-goodbye.md

// rename via hash
gulp.src("./src/main/text/hello.txt", { base: process.cwd() })
  .pipe(rename({
    dirname: "main/text/ciao",
    basename: "aloha",
    prefix: "bonjour-",
    suffix: "-hola",
    extname: ".md"
  }))
  .pipe(gulp.dest("./dist")); // ./dist/main/text/ciao/bonjour-aloha-hola.md
```

**See test/rename.spec.js for more examples and test/path-parsing.spec.js for hairy details.**

## Notes

* `dirname` is the relative path from the base directory set by `gulp.src` to the filename.
  * `gulp.src()` uses glob-stream which sets the base to the parent of the first directory glob (`*`, `**`, [], or extglob). `dirname` is the remaining directories or `./` if none. glob-stream versions >= 3.1.0 (used by gulp >= 3.2.2) accept a `base` option, which can be used to explicitly set the base.
  * `gulp.dest()` renames the directories between `process.cwd()` and `dirname` (i.e. the base relative to CWD). Use `dirname` to rename the directories matched by the glob or descendents of the base of option.
* `basename` is the filename without the extension like path.basename(filename, path.extname(filename)).
* `extname` is the file extension including the '.' like path.extname(filename).

## License

[MIT License](http://en.wikipedia.org/wiki/MIT_License)
