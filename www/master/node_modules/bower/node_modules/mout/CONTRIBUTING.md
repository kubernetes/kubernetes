# Contributing

Fork the repo at https://github.com/mout/mout

 > "Write clearly, don't be too clever" - The Elements of Programming Style

Avoid unnamed functions and follow the other modules structure. By only using named functions it will be easier to extract the code from the AMD module if needed and it will also give better error messages, JavaScript minifiers like [Google Closure Compiler](http://code.google.com/closure/compiler/) and [UglifyJS](https://github.com/mishoo/UglifyJS) will make sure code is as small/optimized as possible.

 > "Make it clear before you make it faster." - The Elements of Programming Style

Be sure to always create tests for each proposed module. Features will only be merged if they contain proper tests and documentation.

 > "Good code is its own best documentation." - Steve McConnell

We should do a code review before merging to make sure names makes sense and implementation is as good as possible.

Try to split your pull requests into logical groups, the smaller the easier to be reviewed/merged.



## Tests & Code Coverage ##

Tests can be found inside the `tests` folder, to execute them in the browser open the `tests/runner.html`. The same tests also work on node.js by running `npm test`.

We should have tests for all methods and ensure we have a high code coverage through our continuous integration server ([travis](https://travis-ci.org/mout/mout)). When you ask for a pull request Travis will automatically run the tests on node.js and check the code coverage as well.

We run `node build pkg` automatically before any `npm test`, so specs and packages should always be in sync. (will avoid human mistakes)

To check code coverage run `npm test --coverage`, it will generate the reports inside the `coverage` folder and also log the results. Please note that node.js doesn't execute all code branches since we have some conditionals that are only met on old JavaScript engines (eg. IE 7-8), so we will never have 100% code coverage (but should be close to it).



## Build Script ##

The [build script](https://github.com/mout/mout/wiki/Build-Script) can be extremely helpful and can avoid human mistakes, use it.



## Admins / Pull Requests ##

Even if you are an admin (have commit rights) please do pull requests when adding new features or changing current behavior, that way we can review the work and discuss. Feel free to push changes that doesn't affect behavior without asking for a pull request (readme, changelog, build script, typos, refactoring, ...).



## Large changes ##

If you are proposing some major change, please create an issue to discuss it first. (maybe it's outside the scope of the project)



## Questions / IRC / Wiki / Issue Tracker ##

When in doubt ask someone on IRC to help you ([#moutjs on irc.freenode.net](http://webchat.freenode.net/?channels=moutjs)) or create a [new issue](http://github.com/mout/mout/issues).

The [project wiki](https://github.com/mout/mout/wiki) can also be a good resource of information.


---

Check the [contributors list at github](https://github.com/mout/mout/contributors).

