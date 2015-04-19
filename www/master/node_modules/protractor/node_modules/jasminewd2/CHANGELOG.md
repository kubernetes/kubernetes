# 1.1.0

## Dependency Updates
- ([daa67d6])(https://github.com/angular/jasminewd/commit/daa67d6eabdd9c70306748da8a0dc0a6f2edb90f)) chore(dependencies): update to selenium-webdriver 2.43.4

# 1.0.4
## Bug Fixes
- ([a088e6f](https://github.com/angular/jasminewd/commit/a088e6f175ca817f59d5eea99549e45ab5861ce0)) fix(timeouts): should call special timeout handlers only for a jasmine timeout

    Previously, it used to call the resets if anything matched 'timeout'. This was too
    vague, since many error messages contain that string.

    Closes #8

# 1.0.3
## Bug Fixes
- ([00821b3](https://github.com/angular/jasminewd/commit/00821b3180a6674012fdccab106835f5ce94bb3f)) fix(timeout): better messaging if the control flow does not have a listed last task

# 1.0.2

## Bug Fixes
- ([30b6811](https://github.com/angular/jasminewd/commit/30b68113759a7cb5c8dabc5b16ffcd89516882d8)) fix(timeout): output more information about the current task when a timeout occurs

# 1.0.1

## Bug Fixes
- ([c507b37](https://github.com/angular/jasminewd/commit/c507b37dd04cf267a437a579fc3b14063abb2ef8))
  fix(index): stop infinite promise resolution

1.0.0
=====

Support for Jasmine 1.3.1. Tested against minijasminenode @ 0.4.0.

Features

 - Automatically makes tests asynchronously wait until the WebDriverJS control flow is empty.

 - If a `done` function is passed to the test, waits for both the control flow and until done is called.

 - Enhances `expect` so that it automatically unwraps promises before performing the assertion.

