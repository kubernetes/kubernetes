#gulp-git
[![Build Status](https://travis-ci.org/stevelacy/gulp-git.png?branch=master)](https://travis-ci.org/stevelacy/gulp-git)
[![NPM version](https://badge.fury.io/js/gulp-git.png)](http://badge.fury.io/js/gulp-git)

<table>
<tr> 
<td>Package</td><td>gulp-git</td>
</tr>
<tr>
<td>Description</td>
<td>Git plugin for Gulp (gulpjs.com)</td>
</tr>
<tr>
<td>Node Version</td>
<td>>= 0.9</td>
</tr>
<tr>
<td>Gulp Version</td>
<td>3.x</td>
</tr>
</table>

## Usage
### Install
    npm install gulp-git --save

#### 0.3.0 introduced Breaking Changes!
Git actions which did not require a [Vinyl](https://github.com/wearefractal/vinyl) file were refactored.
Please review the following docs for changes:
##Example

```javascript
var gulp = require('gulp');
var git = require('gulp-git');

// Run git init 
// src is the root folder for git to initialize
gulp.task('init', function(done){
  git.init({}, done);
});

// Run git init with options
gulp.task('init', function(){
  git.init({args: '--quiet --bare'});
});

// Run git add 
// src is the file(s) to add (or ./*)
gulp.task('add', function(){
  return gulp.src('./git-test/*')
  .pipe(git.add());
});

// Run git add with options
gulp.task('add', function(){
  return gulp.src('./git-test/*')
  .pipe(git.add({args: '-f -i -p'}));
});

// Run git commit
// src are the files to commit (or ./*)
gulp.task('commit', function(){
  return gulp.src('./git-test/*')
  .pipe(git.commit('initial commit'));
});

// Run git commit with options
gulp.task('commit', function(){
  return gulp.src('./git-test/*')
  .pipe(git.commit('initial commit', {args: '-A --amend -s'}));
});

// Run git remote add
// remote is the remote repo
// repo is the https url of the repo
gulp.task('remote', function(done){
  git.addRemote('origin', 'https://github.com/stevelacy/git-test', {}, done);
});

// Run git push 
// remote is the remote repo
// branch is the remote branch to push to
gulp.task('push', function(){
  git.push('origin', 'master');
});

// Run git push with options
// branch is the remote branch to push to
gulp.task('push', function(done){
  git.push('origin', 'master', {args: " -f"}, done);
});

// Run git pull
// remote is the remote repo
// branch is the remote branch to pull from
gulp.task('pull', function(done){
  git.pull('origin', 'master', {args: '--rebase'}, done);
});

// Tag the repo with a version
gulp.task('tag', function(){
  git.tag('v1.1.1', 'Version message');
});

// Tag the repo With signed key
gulp.task('tagsec', function(done){
  git.tag('v1.1.1', 'Version message with signed key', {args: "signed"}, done);
});

// Create a git branch
gulp.task('branch', function(done){
  git.branch('newBranch', {}, done);
});

// Checkout a git branch
gulp.task('checkout', function(){
  return gulp.src('./*')
  .pipe(git.checkout('branchName'));
});

// Merge branches to master
gulp.task('merge', function(done){
  git.merge('branchName', {}, done);
});

// Reset a commit
gulp.task('reset', function(){
  git.reset('SHA');
});

// Git rm a file or folder
gulp.task('rm', function(){
  return gulp.src('./gruntfile.js')
  .pipe(git.rm());
});

// Run gulp's default task
gulp.task('default',['add']);

```

##API

### git.init()
`git init`

Options: Object

`.init({args: 'options'})`

Creates an empty git repo

### git.add()
`git add <files>`

gulp.src: required

Options: Object

`.add({args: 'options'})`

Adds files to repo

### git.commit()
`git commit -m <message> <files>`

gulp.src: required

Options: Object

`.commit('message', {args: 'options'})`

Commits changes to repo

`message` allows templates:

`git.commit('initial commit file: <%= file.path%>');`

### git.addRemote()
`git remote add <remote> <repo https url>`

    defaults:
    remote: 'origin'

Options: Object

`.addRemote('origin', 'git-repo-url', {args: 'options'})`

Adds remote repo url

### git.pull()
`git pull <remote> <branch>`

    defaults:
    remote: 'origin'
    branch: 'master'

Options: Object

`.pull('origin', 'branch', {args: 'options'})`

Pulls changes from remote repo

### git.push()
`git push <remote> <branch>`

    defaults:
    remote: 'origin'
    branch: 'master'

Options: Object

`.push('origin', 'master', {args: 'options'})`

Pushes changes to remote repo

### git.tag()
`git tag -a/s <version> -m <message>`

Options: Object

Tags repo with release version

if options.signed is set to true, the tag will use the git secure key:

`git.tag('v1.1.1', 'Version message with signed key', {signed: true});`


### git.branch()
`git branch <new branch name>`

Options: Object

`.branch('newBranch', {args: "options"})`

Creates a new branch

### git.checkout()
`git checkout <new branch name>`

gulp.src: required

Options: Object

`.checkout('newBranch', {args: "options"})`

Checkouts a new branch with files

### git.merge()
`git merge <branch name> <options>`

Options: Object

`.merge('newBranch', {args: "options"})`

Merges a branch into master

### git.rm()
`git rm <file> <options>`

gulp.src: required

Options: Object

`.rm({args: "options"})`

Removes a file from git and deletes it

### git.reset()
`git reset <SHA> <options>`

Options: Object

`.reset('850f500f53f54', {args: 'options'})`

Resets a git commit

***




####You can view more examples in the [example folder.](https://github.com/stevelacy/gulp-git/tree/master/examples)



## LICENSE

(MIT License)

Copyright (c) 2014 Steve Lacy <me@slacy.me> slacy.me - Fractal <contact@wearefractal.com> wearefractal.com

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
