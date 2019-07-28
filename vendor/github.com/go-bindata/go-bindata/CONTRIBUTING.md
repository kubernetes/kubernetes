## Contribution guidelines.

So you wish to contribute to this project? Fantastic!
Here are a few guidelines to help you do this in a
streamlined fashion.


## Bug reports

When supplying a bug report, please consider the following guidelines.
These serve to make it easier for us to address the issue and find a solution.
Most of these are pretty self-evident, but sometimes it is still necessary
to reiterate them.

* Be clear in the way you express the problem. Use simple language and
  just enough of it to clearly define the issue. Not everyone is a native
  English speaker. And while most can handle themselves pretty well,
  it helps to stay away from more esoteric vocabulary.
  
  Be patient with non-native English speakers. If their bug reports
  or comments are hard to understand, just ask for clarification.
  Do not start guessing at their meaning, as this may just lead to
  more confusion and misunderstandings.
* Clearly define any information which is relevant to the problem.
  This includes library versions, operating system and any other
  external dependencies which may be needed.
* Where applicable, provide a step-by-step listing of the way to
  reproduce the problem. Make sure this is the simplest possible
  way to do so. Omit any and all unneccesary steps, because they may
  just complicate our understanding of the real problem.
  If need be, create a whole new code project on your local machine,
  which specifically tries to create the problem you are running into;
  nothing more, nothing less.
  
  Include this program in the bug report. It often suffices to paste
  the code in a [Gist](https://gist.github.com) or on the
  [Go playground](http://play.golang.org).
* If possible, provide us with a listing of the steps you have already
  undertaken to solve the problem. This can save us a great deal of
  wasted time, trying out solutions you have already covered.


## Pull requests

Bug reports are great. Supplying fixes to bugs is even better.
When submitting a pull request, the following guidelines are
good to keep in mind:

* `go fmt`: **Always** run your code through `go fmt`, before
  committing it. Code has to be readable by many different
  people. And the only way this will be as painless as possible,
  is if we all stick to the same code style.
  
  Some of our projects may have automated build-servers hooked up
  to commit hooks. These will vet any submitted code and determine
  if it meets a set of properties. One of which is code formatting.
  These servers will outright deny a submission which has not been
  run through `go fmt`, even if the code itself is correct.
  
  We try to maintain a zero-tolerance policy on this matter,
  because consistently formatted code makes life a great deal
  easier for everyone involved.
* Commit log messages: When committing changes, do so often and
  clearly -- Even if you have changed only 1 character in a code
  comment. This means that commit log messages should clearly state
  exactly what the change does and why. If it fixes a known issue,
  then mention the issue number in the commit log. E.g.:
  
  > Fixes return value for `foo/boo.Baz()` to be consistent with
  > the rest of the API. This addresses issue #32
  
  Do not pile a lot of unrelated changes into a single commit.
  Pick and choose only those changes for a single commit, which are
  directly related. We would much rather see a hundred commits
  saying nothing but `"Runs go fmt"` in between any real fixes
  than have these style changes embedded in those real fixes.
  It creates a lot of noise when trying to review code.


