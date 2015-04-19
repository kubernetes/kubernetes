2.0.0 / 2014-07-28
------------------
* added `\n` to end of file on write. [#14](https://github.com/jprichardson/node-jsonfile/pull/14)
* added `options.throws` to `readFileSync()`
* dropped support for Node v0.8

1.2.0 / 2014-06-29
------------------
* removed semicolons
* bugfix: passed `options` to `fs.readFile` and `fs.readFileSync`. This technically changes behavior, but 
changes it according to docs. #12

1.1.1 / 2013-11-11
------------------
* fixed catching of callback bug (ffissore / #5)

1.1.0 / 2013-10-11
------------------
* added `options` param to methods, (seanodell / #4)

1.0.1 / 2013-09-05
------------------
* removed `homepage` field from package.json to remove NPM warning

1.0.0 / 2013-06-28
------------------
* added `.npmignore`, #1
* changed spacing default from `4` to `2` to follow Node conventions

0.0.1 / 2012-09-10
------------------
* Initial release.
