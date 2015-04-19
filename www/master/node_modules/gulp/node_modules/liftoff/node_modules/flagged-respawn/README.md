# flagged-respawn [![Build Status](https://secure.travis-ci.org/tkellen/node-flagged-respawn.png)](http://travis-ci.org/tkellen/node-flagged-respawn)
> A tool for respawning node binaries when special flags are present.

[![NPM](https://nodei.co/npm/flagged-respawn.png)](https://nodei.co/npm/flagged-respawn/)

## What is it?

Say you wrote a command line tool that runs arbitrary javascript (e.g. task runner, test framework, etc). For the sake of discussion, let's pretend it's a testing harness you've named `testify`.

Everything is going splendidly until one day you decide to test some code that relies on a feature behind a v8 flag in node (`--harmony`, for example).  Without much thought, you run `testify --harmony spec tests.js`.

It doesn't work. After digging around for a bit, you realize this produces a [`process.argv`](http://nodejs.org/docs/latest/api/process.html#process_process_argv) of:

`['node', '/usr/local/bin/test', '--harmony', 'spec', 'tests.js']`

Crap. The `--harmony` flag is in the wrong place! It should be applied to the **node** command, not our binary. What we actually wanted was this:

`['node', '--harmony', '/usr/local/bin/test', 'spec', 'tests.js']`

Flagged-respawn fixes this problem and handles all the edge cases respawning creates, such as:
- Providing a method to determine if a respawn is needed.
- Piping stderr/stdout from the child into the parent.
- Making the parent process exit with the same code as the child.
- If the child is killed, making the parent exit with the same signal.

To see it in action, clone this repository and run `npm install` / `npm run respawn` / `npm run nospawn`.

## Sample Usage

```js
#!/usr/bin/env node

const flaggedRespawn = require('flagged-respawn');

// get a list of all possible v8 flags for the running version of node
const v8flags = require('v8flags').fetch();

flaggedRespawn(v8flags, process.argv, function (ready, child) {
  if (ready) {
    console.log('Running!');
    // your cli code here
  } else {
    console.log('Special flags found, respawning.');
  }
  if (process.pid !== child.pid) {
    console.log('Respawned to PID:', child.pid);
  }
});

```

## Release History

* 2014-09-12 - v0.3.1 - use `{ stdio: 'inherit' }` for spawn to maintain colors
* 2014-09-11 - v0.3.0 - for real this time
* 2014-09-11 - v0.2.0 - cleanup
* 2014-09-04 - v0.1.1 - initial release
