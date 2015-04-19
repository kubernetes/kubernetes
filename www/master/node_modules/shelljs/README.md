# ShellJS - Unix shell commands for Node.js [![Build Status](https://secure.travis-ci.org/arturadib/shelljs.png)](http://travis-ci.org/arturadib/shelljs)

ShellJS is a portable **(Windows/Linux/OS X)** implementation of Unix shell commands on top of the Node.js API. You can use it to eliminate your shell script's dependency on Unix while still keeping its familiar and powerful commands. You can also install it globally so you can run it from outside Node projects - say goodbye to those gnarly Bash scripts!

The project is [unit-tested](http://travis-ci.org/arturadib/shelljs) and battled-tested in projects like:

+ [PDF.js](http://github.com/mozilla/pdf.js) - Firefox's next-gen PDF reader
+ [Firebug](http://getfirebug.com/) - Firefox's infamous debugger
+ [JSHint](http://jshint.com) - Most popular JavaScript linter
+ [Zepto](http://zeptojs.com) - jQuery-compatible JavaScript library for modern browsers
+ [Yeoman](http://yeoman.io/) - Web application stack and development tool
+ [Deployd.com](http://deployd.com) - Open source PaaS for quick API backend generation

and [many more](https://npmjs.org/browse/depended/shelljs).

## Installing

Via npm:

```bash
$ npm install [-g] shelljs
```

If the global option `-g` is specified, the binary `shjs` will be installed. This makes it possible to
run ShellJS scripts much like any shell script from the command line, i.e. without requiring a `node_modules` folder:

```bash
$ shjs my_script
```

You can also just copy `shell.js` into your project's directory, and `require()` accordingly.


## Examples

### JavaScript

```javascript
require('shelljs/global');

if (!which('git')) {
  echo('Sorry, this script requires git');
  exit(1);
}

// Copy files to release dir
mkdir('-p', 'out/Release');
cp('-R', 'stuff/*', 'out/Release');

// Replace macros in each .js file
cd('lib');
ls('*.js').forEach(function(file) {
  sed('-i', 'BUILD_VERSION', 'v0.1.2', file);
  sed('-i', /.*REMOVE_THIS_LINE.*\n/, '', file);
  sed('-i', /.*REPLACE_LINE_WITH_MACRO.*\n/, cat('macro.js'), file);
});
cd('..');

// Run external tool synchronously
if (exec('git commit -am "Auto-commit"').code !== 0) {
  echo('Error: Git commit failed');
  exit(1);
}
```

### CoffeeScript

```coffeescript
require 'shelljs/global'

if not which 'git'
  echo 'Sorry, this script requires git'
  exit 1

# Copy files to release dir
mkdir '-p', 'out/Release'
cp '-R', 'stuff/*', 'out/Release'

# Replace macros in each .js file
cd 'lib'
for file in ls '*.js'
  sed '-i', 'BUILD_VERSION', 'v0.1.2', file
  sed '-i', /.*REMOVE_THIS_LINE.*\n/, '', file
  sed '-i', /.*REPLACE_LINE_WITH_MACRO.*\n/, cat 'macro.js', file
cd '..'

# Run external tool synchronously
if (exec 'git commit -am "Auto-commit"').code != 0
  echo 'Error: Git commit failed'
  exit 1
```

## Global vs. Local

The example above uses the convenience script `shelljs/global` to reduce verbosity. If polluting your global namespace is not desirable, simply require `shelljs`.

Example:

```javascript
var shell = require('shelljs');
shell.echo('hello world');
```

## Make tool

A convenience script `shelljs/make` is also provided to mimic the behavior of a Unix Makefile. In this case all shell objects are global, and command line arguments will cause the script to execute only the corresponding function in the global `target` object. To avoid redundant calls, target functions are executed only once per script.

Example (CoffeeScript):

```coffeescript
require 'shelljs/make'

target.all = ->
  target.bundle()
  target.docs()

target.bundle = ->
  cd __dirname
  mkdir 'build'
  cd 'lib'
  (cat '*.js').to '../build/output.js'

target.docs = ->
  cd __dirname
  mkdir 'docs'
  cd 'lib'
  for file in ls '*.js'
    text = grep '//@', file     # extract special comments
    text.replace '//@', ''      # remove comment tags
    text.to 'docs/my_docs.md'
```

To run the target `all`, call the above script without arguments: `$ node make`. To run the target `docs`: `$ node make docs`, and so on.



<!-- 

  DO NOT MODIFY BEYOND THIS POINT - IT'S AUTOMATICALLY GENERATED

-->


## Command reference


All commands run synchronously, unless otherwise stated.


### cd('dir')
Changes to directory `dir` for the duration of the script


### pwd()
Returns the current directory.


### ls([options ,] path [,path ...])
### ls([options ,] path_array)
Available options:

+ `-R`: recursive
+ `-A`: all files (include files beginning with `.`, except for `.` and `..`)

Examples:

```javascript
ls('projs/*.js');
ls('-R', '/users/me', '/tmp');
ls('-R', ['/users/me', '/tmp']); // same as above
```

Returns array of files in the given path, or in current directory if no path provided.


### find(path [,path ...])
### find(path_array)
Examples:

```javascript
find('src', 'lib');
find(['src', 'lib']); // same as above
find('.').filter(function(file) { return file.match(/\.js$/); });
```

Returns array of all files (however deep) in the given paths.

The main difference from `ls('-R', path)` is that the resulting file names
include the base directories, e.g. `lib/resources/file1` instead of just `file1`.


### cp([options ,] source [,source ...], dest)
### cp([options ,] source_array, dest)
Available options:

+ `-f`: force
+ `-r, -R`: recursive

Examples:

```javascript
cp('file1', 'dir1');
cp('-Rf', '/tmp/*', '/usr/local/*', '/home/tmp');
cp('-Rf', ['/tmp/*', '/usr/local/*'], '/home/tmp'); // same as above
```

Copies files. The wildcard `*` is accepted.


### rm([options ,] file [, file ...])
### rm([options ,] file_array)
Available options:

+ `-f`: force
+ `-r, -R`: recursive

Examples:

```javascript
rm('-rf', '/tmp/*');
rm('some_file.txt', 'another_file.txt');
rm(['some_file.txt', 'another_file.txt']); // same as above
```

Removes files. The wildcard `*` is accepted.


### mv(source [, source ...], dest')
### mv(source_array, dest')
Available options:

+ `f`: force

Examples:

```javascript
mv('-f', 'file', 'dir/');
mv('file1', 'file2', 'dir/');
mv(['file1', 'file2'], 'dir/'); // same as above
```

Moves files. The wildcard `*` is accepted.


### mkdir([options ,] dir [, dir ...])
### mkdir([options ,] dir_array)
Available options:

+ `p`: full path (will create intermediate dirs if necessary)

Examples:

```javascript
mkdir('-p', '/tmp/a/b/c/d', '/tmp/e/f/g');
mkdir('-p', ['/tmp/a/b/c/d', '/tmp/e/f/g']); // same as above
```

Creates directories.


### test(expression)
Available expression primaries:

+ `'-b', 'path'`: true if path is a block device
+ `'-c', 'path'`: true if path is a character device
+ `'-d', 'path'`: true if path is a directory
+ `'-e', 'path'`: true if path exists
+ `'-f', 'path'`: true if path is a regular file
+ `'-L', 'path'`: true if path is a symboilc link
+ `'-p', 'path'`: true if path is a pipe (FIFO)
+ `'-S', 'path'`: true if path is a socket

Examples:

```javascript
if (test('-d', path)) { /* do something with dir */ };
if (!test('-f', path)) continue; // skip if it's a regular file
```

Evaluates expression using the available primaries and returns corresponding value.


### cat(file [, file ...])
### cat(file_array)

Examples:

```javascript
var str = cat('file*.txt');
var str = cat('file1', 'file2');
var str = cat(['file1', 'file2']); // same as above
```

Returns a string containing the given file, or a concatenated string
containing the files if more than one file is given (a new line character is
introduced between each file). Wildcard `*` accepted.


### 'string'.to(file)

Examples:

```javascript
cat('input.txt').to('output.txt');
```

Analogous to the redirection operator `>` in Unix, but works with JavaScript strings (such as
those returned by `cat`, `grep`, etc). _Like Unix redirections, `to()` will overwrite any existing file!_


### 'string'.toEnd(file)

Examples:

```javascript
cat('input.txt').toEnd('output.txt');
```

Analogous to the redirect-and-append operator `>>` in Unix, but works with JavaScript strings (such as
those returned by `cat`, `grep`, etc).


### sed([options ,] search_regex, replace_str, file)
Available options:

+ `-i`: Replace contents of 'file' in-place. _Note that no backups will be created!_

Examples:

```javascript
sed('-i', 'PROGRAM_VERSION', 'v0.1.3', 'source.js');
sed(/.*DELETE_THIS_LINE.*\n/, '', 'source.js');
```

Reads an input string from `file` and performs a JavaScript `replace()` on the input
using the given search regex and replacement string. Returns the new string after replacement.


### grep([options ,] regex_filter, file [, file ...])
### grep([options ,] regex_filter, file_array)
Available options:

+ `-v`: Inverse the sense of the regex and print the lines not matching the criteria.

Examples:

```javascript
grep('-v', 'GLOBAL_VARIABLE', '*.js');
grep('GLOBAL_VARIABLE', '*.js');
```

Reads input string from given files and returns a string containing all lines of the
file that match the given `regex_filter`. Wildcard `*` accepted.


### which(command)

Examples:

```javascript
var nodeExec = which('node');
```

Searches for `command` in the system's PATH. On Windows looks for `.exe`, `.cmd`, and `.bat` extensions.
Returns string containing the absolute path to the command.


### echo(string [,string ...])

Examples:

```javascript
echo('hello world');
var str = echo('hello world');
```

Prints string to stdout, and returns string with additional utility methods
like `.to()`.


### pushd([options,] [dir | '-N' | '+N'])

Available options:

+ `-n`: Suppresses the normal change of directory when adding directories to the stack, so that only the stack is manipulated.

Arguments:

+ `dir`: Makes the current working directory be the top of the stack, and then executes the equivalent of `cd dir`.
+ `+N`: Brings the Nth directory (counting from the left of the list printed by dirs, starting with zero) to the top of the list by rotating the stack.
+ `-N`: Brings the Nth directory (counting from the right of the list printed by dirs, starting with zero) to the top of the list by rotating the stack.

Examples:

```javascript
// process.cwd() === '/usr'
pushd('/etc'); // Returns /etc /usr
pushd('+1');   // Returns /usr /etc
```

Save the current directory on the top of the directory stack and then cd to `dir`. With no arguments, pushd exchanges the top two directories. Returns an array of paths in the stack.

### popd([options,] ['-N' | '+N'])

Available options:

+ `-n`: Suppresses the normal change of directory when removing directories from the stack, so that only the stack is manipulated.

Arguments:

+ `+N`: Removes the Nth directory (counting from the left of the list printed by dirs), starting with zero.
+ `-N`: Removes the Nth directory (counting from the right of the list printed by dirs), starting with zero.

Examples:

```javascript
echo(process.cwd()); // '/usr'
pushd('/etc');       // '/etc /usr'
echo(process.cwd()); // '/etc'
popd();              // '/usr'
echo(process.cwd()); // '/usr'
```

When no arguments are given, popd removes the top directory from the stack and performs a cd to the new top directory. The elements are numbered from 0 starting at the first directory listed with dirs; i.e., popd is equivalent to popd +0. Returns an array of paths in the stack.

### dirs([options | '+N' | '-N'])

Available options:

+ `-c`: Clears the directory stack by deleting all of the elements.

Arguments:

+ `+N`: Displays the Nth directory (counting from the left of the list printed by dirs when invoked without options), starting with zero.
+ `-N`: Displays the Nth directory (counting from the right of the list printed by dirs when invoked without options), starting with zero.

Display the list of currently remembered directories. Returns an array of paths in the stack, or a single path if +N or -N was specified.

See also: pushd, popd


### exit(code)
Exits the current process with the given exit code.

### env['VAR_NAME']
Object containing environment variables (both getter and setter). Shortcut to process.env.

### exec(command [, options] [, callback])
Available options (all `false` by default):

+ `async`: Asynchronous execution. Defaults to true if a callback is provided.
+ `silent`: Do not echo program output to console.

Examples:

```javascript
var version = exec('node --version', {silent:true}).output;

var child = exec('some_long_running_process', {async:true});
child.stdout.on('data', function(data) {
  /* ... do something with data ... */
});

exec('some_long_running_process', function(code, output) {
  console.log('Exit code:', code);
  console.log('Program output:', output);
});
```

Executes the given `command` _synchronously_, unless otherwise specified.
When in synchronous mode returns the object `{ code:..., output:... }`, containing the program's
`output` (stdout + stderr)  and its exit `code`. Otherwise returns the child process object, and
the `callback` gets the arguments `(code, output)`.

**Note:** For long-lived processes, it's best to run `exec()` asynchronously as
the current synchronous implementation uses a lot of CPU. This should be getting
fixed soon.


### chmod(octal_mode || octal_string, file)
### chmod(symbolic_mode, file)

Available options:

+ `-v`: output a diagnostic for every file processed
+ `-c`: like verbose but report only when a change is made
+ `-R`: change files and directories recursively

Examples:

```javascript
chmod(755, '/Users/brandon');
chmod('755', '/Users/brandon'); // same as above
chmod('u+x', '/Users/brandon');
```

Alters the permissions of a file or directory by either specifying the
absolute permissions in octal form or expressing the changes in symbols.
This command tries to mimic the POSIX behavior as much as possible.
Notable exceptions:

+ In symbolic modes, 'a-r' and '-r' are identical.  No consideration is
  given to the umask.
+ There is no "quiet" option since default behavior is to run silent.


## Non-Unix commands


### tempdir()

Examples:

```javascript
var tmp = tempdir(); // "/tmp" for most *nix platforms
```

Searches and returns string containing a writeable, platform-dependent temporary directory.
Follows Python's [tempfile algorithm](http://docs.python.org/library/tempfile.html#tempfile.tempdir).


### error()
Tests if error occurred in the last command. Returns `null` if no error occurred,
otherwise returns string explaining the error


## Configuration


### config.silent
Example:

```javascript
var silentState = config.silent; // save old silent state
config.silent = true;
/* ... */
config.silent = silentState; // restore old silent state
```

Suppresses all command output if `true`, except for `echo()` calls.
Default is `false`.

### config.fatal
Example:

```javascript
config.fatal = true;
cp('this_file_does_not_exist', '/dev/null'); // dies here
/* more commands... */
```

If `true` the script will die on errors. Default is `false`.
