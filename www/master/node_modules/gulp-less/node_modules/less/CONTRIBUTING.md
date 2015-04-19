# Contributing to Less.js

> We welcome feature requests and bug reports. Please read these guidelines before submitting one. 


<span class="warning">**Words that begin with the at sign (`@`) must be wrapped in backticks!** </span>. As a courtesy to avoid sending notifications to any user that might have the `@username` being referenced, please remember that GitHub usernames also start with the at sign. If you don't wrap them in backticks, users will get unintended notifications from you.

GitHub has other great markdown features as well, [go here to learn more about them](https://help.github.com/articles/github-flavored-markdown).   


## Reporting Issues

We only accept issues that are bug reports or feature requests. Bugs must be isolated and reproducible problems that we can fix within the Less.js core. Please read the following guidelines before opening any issue. 

1. **Search for existing issues.** We get a lot of duplicate issues, and you'd help us out a lot by first checking if someone else has reported the same issue. Moreover, the issue may have already been resolved with a fix available.
2. **Create an isolated and reproducible test case.** Be sure the problem exists in Less.js's code with [reduced test cases](http://css-tricks.com/reduced-test-cases/) that should be included in each bug report.
3. **Test with the latest version**. We get a lot of issues that could be resolved by updating your version of Less.js. 
3. **Include an example with source.** E.g. You can use [less2css.org](http://less2css.org/) to create a short test case. 
4. **Share as much information as possible.** Include operating system and version. Describe how you use Less. If you use it in the browser, please include browser and version, and the version of Less.js you're using. Let us know if you're using the command line (`lessc`) or an external tool. And try to include steps to reproduce the bug.
5. If you have a solution or suggestion for how to fix the bug you're reporting, please include it, or make a pull request - don't assume the maintainers know how to fix it just because you do.

Please report documentation issues in [the documentation project](https://github.com/less/less-docs).

## Feature Requests

* Please search for existing feature requests first to see if something similar already exists.
* Include a clear and specific use-case. We love new ideas, but we do not add language features without a reason.
* Consider whether or not your language feature would be better as a function or implemented in a 3rd-party build system such as [assemble-less](http://github.com/assemble/assemble-less).


## Pull Requests

_Pull requests are encouraged!_

* Start by adding a feature request to get feedback and see how your idea is received. 
* If your pull request solves an existing issue, but it's different in some way, _please create a new issue_ and make sure to discuss it with the core contributors. Otherwise you risk your hard work being rejected.
* Do not change the **./dist/** folder, we do this when releasing
* _Please add tests_ for your work. Tests are invoked using `grunt test` command. It will run both node.js tests and browser ([PhantomJS](http://phantomjs.org/)) tests. 

### Coding Standards

* Always use spaces, never tabs
* End lines in semi-colons. 
* Loosely aim towards jsHint standards


## Developing
If you want to take an issue just add a small comment saying you are having a go at something, so we don't get duplication.

Learn more about [developing Less.js](http://lesscss.org/usage/#developing-less).
