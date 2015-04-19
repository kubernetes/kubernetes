gulp-tag-version
================

Tag git repository with current package version (gulp plugin).

It will read the `version` attribute (by default, override with `key` option) from the JSON stream (probably your `package.json` or `bower.json`), prefixes it with `"v"` (override with `prefix` option) and _tags_ the repository (effectively issues the `git tag <tagname>` command) with such created tagname (e.g. `v1.2.3`).


Simple example gulpfile
-----------------------
```js
var gulp = require('gulp'),
    tag_version = require('gulp-tag-version');

// Assuming there's "version: 1.2.3" in package.json,
// tag the last commit as "v1.2.3"//
gulp.task('tag', function() {
  return gulp.src(['./package.json']).pipe(tag_version());
});
```


Advanced example gulpfile (with bumping and commiting)
------------------------------------------------------

```js

// dependencies
var gulp = require('gulp'),
    git = require('gulp-git'),
    bump = require('gulp-bump'),
    filter = require('gulp-filter'),
    tag_version = require('gulp-tag-version');

/**
 * Bumping version number and tagging the repository with it.
 * Please read http://semver.org/
 *
 * You can use the commands
 *
 *     gulp patch     # makes v0.1.0 → v0.1.1
 *     gulp feature   # makes v0.1.1 → v0.2.0
 *     gulp release   # makes v0.2.1 → v1.0.0
 *
 * To bump the version numbers accordingly after you did a patch,
 * introduced a feature or made a backwards-incompatible release.
 */

function inc(importance) {
    // get all the files to bump version in
    return gulp.src(['./package.json', './bower.json'])
        // bump the version number in those files
        .pipe(bump({type: importance}))
        // save it back to filesystem
        .pipe(gulp.dest('./'))
        // commit the changed version number
        .pipe(git.commit('bumps package version'))

        // read only one file to get the version number
        .pipe(filter('package.json'))
        // **tag it in the repository**
        .pipe(tag_version());
}

gulp.task('patch', function() { return inc('patch'); })
gulp.task('feature', function() { return inc('minor'); })
gulp.task('release', function() { return inc('major'); })
```

Other features/remarks
----------------------

* If you need any special tagging options to be passed down to `git.tag`, just add it to the `tag_version` options. For example:
```js
gulp.task('bump_submodule', function(){
    return gulp.src('./bower.json',  { cwd: './dist' })
        .pipe(bump({type: 'patch'}))
        .pipe(gulp.dest('./',{ cwd: './dist' }))
        .pipe(git.commit('bumps package version',{cwd: './dist'}))
        .pipe(filter('bower.json'))
        .pipe(tag_version({cwd: './dist'}));
});
```

* If you don't want the version number to be read from the input stream, use the `version` parameter:
```js*
return gulp.src ...
  ...
  .pipe(tag_version({version: '1.2.3'}));
```

Thanks :beer:
--------

* to guys and gals from Fractal for [Gulp](http://gulpjs.com/) itself, obviously
* to Steve Lacy (http://slacy.me) for creating [`gulp-bump`](https://github.com/stevelacy/gulp-bump) and [`gulp-git`](https://github.com/stevelacy/gulp-git) used here
* The main file structure is based on `gulp-bump` a bit as well (this is my first plugin :))
* To [@pacemkr](https://github.com/pacemkr) for the first pull request I ever got (supporting empty prefix)
* To [@lapanoid](https://github.com/lapanoid) for passing `opts` down to `git.tag`
* To [@brianmhunt](https://github.com/brianmhunt) for suggesting the `version` parameter
