# Developing for Jasmine Core

We welcome your contributions - Thanks for helping make Jasmine a better project for everyone. Please review the backlog and discussion lists (the main group - [http://groups.google.com/group/jasmine-js](http://groups.google.com/group/jasmine-js) and the developer's list - [http://groups.google.com/group/jasmine-js-dev](http://groups.google.com/group/jasmine-js-dev)) before starting work - what you're looking for may already have been done. If it hasn't, the community can help make your contribution better.

## General Workflow

Please submit pull requests via feature branches using the semi-standard workflow of:

1. Fork it
1. Clone your fork: (`git clone git@github.com:yourUserName/jasmine.git`)                                                                                                                         
1. Change directory: (`cd jasmine`)                                                                                                                                                               
1. Asign original repository to a remote named 'upstream': (`git remote add                                                                                                                       
upstream https://github.com/pivotal/jasmine.git`)                                                                                                                                                 
1. Pull in changes not present in your local repository: (`git fetch upstream`)
1. Create your feature branch (`git checkout -b my-new-feature`)
1. Commit your changes (`git commit -am 'Add some feature'`)
1. Push to the branch (`git push origin my-new-feature`)
1. Create new Pull Request

We favor pull requests with very small, single commits with a single purpose.

## Background

### Directory Structure

* `/src` contains all of the source files
    * `/src/console` - Node.js-specific files
    * `/src/core` - generic source files
    * `/src/html` - browser-specific files
* `/spec` contains all of the tests
    * mirrors the source directory
    * there are some additional files
* `/dist` contains the standalone distributions as zip files
* `/lib` contains the generated files for distribution as the Jasmine Rubygem and the Python package

### Self-testing

Note that Jasmine tests itself. The files in `lib` are loaded first, defining the reference `jasmine`. Then the files in `src` are loaded, defining the reference `j$`. So there are two copies of the code loaded under test.

The tests should always use `j$` to refer to the objects and functions that are being tested. But the tests can use functions on `jasmine` as needed. _Be careful how you structure any new test code_. Copy the patterns you see in the existing code - this ensures that the code you're testing is not leaking into the `jasmine` reference and vice-versa.

### `boot.js`

__This is new for Jasmine 2.0.__

This file does all of the setup necessary for Jasmine to work. It loads all of the code, creates an `Env`, attaches the global functions, and builds the reporter. It also sets up the execution of the `Env` - for browsers this is in `window.onload`. While the default in `lib` is appropriate for browsers, projects may wish to customize this file.

For example, for Jasmine development there is a different `dev_boot.js` for Jasmine development that does more work.

### Compatibility

* Browser Minimum
  * IE8
  * Firefox 3.x
  * Chrome ??
  * Safari 5

## Development

All source code belongs in `src/`. The `core/` directory contains the bulk of Jasmine's functionality. This code should remain browser- and environment-agnostic. If your feature or fix cannot be, as mentioned above, please degrade gracefully. Any code that should only be in a non-browser environment should live in `src/console/`. Any code that depends on a browser (specifically, it expects `window` to be the global or `document` is present) should live in `src/html/`.

### Install Dependencies

Jasmine Core relies on Ruby and Node.js.

To install the Ruby dependencies, you will need Ruby, Rubygems, and Bundler available. Then:

    $ bundle

...will install all of the Ruby dependencies.

To install the Node dependencies, you will need Node.js, Npm, and [Grunt](http://gruntjs.com/), the [grunt-cli](https://github.com/gruntjs/grunt-cli) and ensure that `grunt` is on your path.

    $ npm install --local

...will install all of the node modules locally. If when you run

    $ grunt

...you see that JSHint runs your system is ready.

### How to write new Jasmine code

Or, How to make a successful pull request

* _Do not change the public interface_. Lots of projects depend on Jasmine and if you aren't careful you'll break them
* _Be environment agnostic_ - server-side developers are just as important as browser developers
* _Be browser agnostic_ - if you must rely on browser-specific functionality, please write it in a way that degrades gracefully
* _Write specs_ - Jasmine's a testing framework; don't add functionality without test-driving it
* _Write code in the style of the rest of the repo_ - Jasmine should look like a cohesive whole
* _Ensure the *entire* test suite is green_ in all the big browsers, Node, and JSHint - your contribution shouldn't break Jasmine for other users

Follow these tips and your pull request, patch, or suggestion is much more likely to be integrated.

### Running Specs

Jasmine uses the [Jasmine Ruby gem](http://github.com/pivotal/jasmine-gem) to test itself in browser.

    $ rake jasmine

...and then visit `http://localhost:8888` to run specs.

Jasmine uses the [Jasmine NPM package](http://github.com/pivotal/jasmine-npm) to test itself in a Node.js/npm environment.

    $ grunt execSpecsInNode

...and then the results will print to the console. All specs run except those that expect a browser (the specs in `spec/html` are ignored).

## Before Committing or Submitting a Pull Request

1. Ensure all specs are green in browser *and* node
1. Ensure JSHint is green with `grunt jshint`
1. Build `jasmine.js` with `grunt buildDistribution` and run all specs again - this ensures that your changes self-test well

## Submitting a Pull Request
1. Revert your changes to `jasmine.js` and `jasmine-html.js`
  * We do this because `jasmine.js` and `jasmine-html.js` are auto-generated (as you've seen in the previous steps) and accepting multiple pull requests when this auto-generated file changes causes lots of headaches.
1. When we accept your pull request, we will generate these files as a separate commit and merge the entire branch into master.

Note that we use Travis for Continuous Integration. We only accept green pull requests.

