# Contributing to the Google API Go Client

## Master git repo

Our master git repo is https://code.googlesource.com/google-api-go-client

## Pull Requests

We do **NOT** use Github pull requests. We use Gerrit instead
with the same workflow as Go. See below.

## The source tree

Most of this project is auto-generated.

The notable directories which are not auto-generated:

```
   google-api-go-generator/ -- the generator itself
   googleapi/               -- shared common code, used by auto-generated code
   examples/                -- sample code
```

# Contribution Guidelines

## Introduction

This document explains how to contribute changes to the google-api-go-client project.

## Testing redux

You've written and tested your code, but
before sending code out for review, run all the tests for the whole
tree to make sure the changes don't break other packages or programs:

```
$ make cached
$ go test ./...
...
ok  	google.golang.org/api/google-api-go-generator	0.226s
ok  	google.golang.org/api/googleapi	0.015s
...
```

Ideally, you will add unit tests to one of the above directories to
demonstrate the changes you are making and include the tests with your
code review.

## Code review

Changes to google-api-go-client must be reviewed before they are submitted,
no matter who makes the change.
A custom git command called `git-codereview`,
discussed below, helps manage the code review process through a Google-hosted
[instance](https://code-review.googlesource.com/) of the code review
system called [Gerrit](https://code.google.com/p/gerrit/).

### Set up authentication for code review

The Git code hosting server and Gerrit code review server both use a Google
Account to authenticate. You therefore need a Google Account to proceed.
(If you can use the account to
[sign in at google.com](https://www.google.com/accounts/Login),
you can use it to sign in to the code review server.)
The email address you use with the code review system
needs to be added to the [`CONTRIBUTORS`](/CONTRIBUTORS) file
with your first code review.
You can [create a Google Account](https://www.google.com/accounts/NewAccount)
associated with any address where you receive email.

Visit the site [code.googlesource.com](https://code.googlesource.com)
and log in using your Google Account.
Click on the "Generate Password" link that appears at the top of the page.

Click the radio button that says "Only `code.googlesource.com`"
to use this authentication token only for the google-api-go-client project.

Further down the page is a box containing commands to install
the authentication cookie in file called `.gitcookies` in your home
directory.
Copy the text for the commands into a Unix shell window to execute it.
That will install the authentication token.

(If you are on a Windows computer, you should instead follow the instructions
in the yellow box to run the command.)

### Register with Gerrit

Now that you have a Google account and the authentication token,
you need to register your account with Gerrit, the code review system.
To do this, visit [golang.org/cl](https://golang.org/cl)
and log in using the same Google Account you used above.
That is all that is required.

### Install the git-codereview command

Now install the `git-codereview` command by running,

```
go get -u golang.org/x/review/git-codereview
```

Make sure `git-codereview` is installed in your shell path, so that the
`git` command can find it. Check that

```
$ git codereview help
```

prints help text, not an error.

Note to Git aficionados: The `git-codereview` command is not required to
upload and manage Gerrit code reviews. For those who prefer plain Git, the text
below gives the Git equivalent of each git-codereview command. If you do use plain
Git, note that you still need the commit hooks that the git-codereview command
configures; those hooks add a Gerrit `Change-Id` line to the commit
message and check that all Go source files have been formatted with gofmt. Even
if you intend to use plain Git for daily work, install the hooks in a new Git
checkout by running `git-codereview hooks`.

### Set up git aliases

The `git-codereview` command can be run directly from the shell
by typing, for instance,

```
$ git codereview sync
```

but it is more convenient to set up aliases for `git-codereview`'s own
subcommands, so that the above becomes,

```
$ git sync
```

The `git-codereview` subcommands have been chosen to be distinct from
Git's own, so it's safe to do so.

The aliases are optional, but in the rest of this document we will assume
they are installed.
To install them, copy this text into your Git configuration file
(usually `.gitconfig` in your home directory):

```
[alias]
	change = codereview change
	gofmt = codereview gofmt
	mail = codereview mail
	pending = codereview pending
	submit = codereview submit
	sync = codereview sync
```

### Understanding the git-codereview command

After installing the `git-codereview` command, you can run

```
$ git codereview help
```

to learn more about its commands.
You can also read the [command documentation](https://godoc.org/golang.org/x/review/git-codereview).

### Switch to the master branch

New changes should
only be made based on the master branch.
Before making a change, make sure you start on the master branch:

```
$ git checkout master
$ git sync
````

(In Git terms, `git sync` runs
`git pull -r`.)

### Make a change

The entire checked-out tree is writable.
Once you have edited files, you must tell Git that they have been modified.
You must also tell Git about any files that are added, removed, or renamed files.
These operations are done with the usual Git commands,
`git add`,
`git rm`,
and
`git mv`.

If you wish to checkpoint your work, or are ready to send the code out for review, run

```
$ git change <branch>
```

from any directory in your google-api-go-client repository to commit the changes so far.
The name `<branch>` is an arbitrary one you choose to identify the
local branch containing your changes.

(In Git terms, `git change <branch>`
runs `git checkout -b branch`,
then `git branch --set-upstream-to origin/master`,
then `git commit`.)

Git will open a change description file in your editor.
(It uses the editor named by the `$EDITOR` environment variable,
`vi` by default.)
The file will look like:

```
# Please enter the commit message for your changes. Lines starting
# with '#' will be ignored, and an empty message aborts the commit.
# On branch foo
# Changes not staged for commit:
#	modified:   editedfile.go
#
```

At the beginning of this file is a blank line; replace it
with a thorough description of your change.
The first line of the change description is conventionally a one-line
summary of the change, prefixed by `google-api-go-client:`,
and is used as the subject for code review mail.
The rest of the
description elaborates and should provide context for the
change and explain what it does.
If there is a helpful reference, mention it here.

After editing, the template might now read:

```
math: improved Sin, Cos and Tan precision for very large arguments

The existing implementation has poor numerical properties for
large arguments, so use the McGillicutty algorithm to improve
accuracy above 1e10.

The algorithm is described at http://wikipedia.org/wiki/McGillicutty_Algorithm

Fixes #54

# Please enter the commit message for your changes. Lines starting
# with '#' will be ignored, and an empty message aborts the commit.
# On branch foo
# Changes not staged for commit:
#	modified:   editedfile.go
#
```

The commented section of the file lists all the modified files in your client.
It is best to keep unrelated changes in different change lists,
so if you see a file listed that should not be included, abort
the command and move that file to a different branch.

The special notation "Fixes #54" associates the change with issue 54 in the
[google-api-go-client issue tracker](https://github.com/google/google-api-go-client/issues/54).
When this change is eventually submitted, the issue
tracker will automatically mark the issue as fixed.
(There are several such conventions, described in detail in the
[GitHub Issue Tracker documentation](https://help.github.com/articles/closing-issues-via-commit-messages/).)

Once you have finished writing the commit message,
save the file and exit the editor.

If you wish to do more editing, re-stage your changes using
`git add`, and then run

```
$ git change
```

to update the change description and incorporate the staged changes.  The
change description contains a `Change-Id` line near the bottom,
added by a Git commit hook during the initial
`git change`.
That line is used by Gerrit to match successive uploads of the same change.
Do not edit or delete it.

(In Git terms, `git change` with no branch name
runs `git commit --amend`.)

### Mail the change for review

Once the change is ready, mail it out for review:

```
$ git mail
```

You can specify a reviewer or CC interested parties
using the `-r` or `-cc` options.
Both accept a comma-separated list of email addresses:

```
$ git mail -r joe@golang.org -cc mabel@example.com,math-nuts@swtch.com
```

Unless explicitly told otherwise, such as in the discussion leading
up to sending in the change list, please specify
`bradfitz@golang.org`, `gmlewis@google.com`, or
`mcgreevy@golang.org` as a reviewer.

(In Git terms, `git mail` pushes the local committed
changes to Gerrit using `git push origin HEAD:refs/for/master`.)

If your change relates to an open issue, please add a comment to the issue
announcing your proposed fix, including a link to your CL.

The code review server assigns your change an issue number and URL,
which `git mail` will print, something like:

```
remote: New Changes:
remote:   https://code-review.googlesource.com/99999 math: improved Sin, Cos and Tan precision for very large arguments
```

### Reviewing code

Running `git mail` will send an email to you and the
reviewers asking them to visit the issue's URL and make comments on the change.
When done, the reviewer adds comments through the Gerrit user interface
and clicks "Reply" to send comments back.
You will receive a mail notification when this happens.
You must reply through the web interface.

### Revise and upload

You must respond to review comments through the web interface.

When you have revised the code and are ready for another round of review,
stage those changes and use `git change` to update the
commit.
To send the update change list for another round of review,
run `git mail` again.

The reviewer can comment on the new copy, and the process repeats.
The reviewer approves the change by giving it a positive score
(+1 or +2) and replying `LGTM`: looks good to me.

You can see a list of your pending changes by running
`git pending`, and switch between change branches with
`git change <branch>`.

### Synchronize your client

While you were working, others might have submitted changes to the repository.
To update your local branch, run

```
$ git sync
```

(In git terms, `git sync` runs
`git pull -r`.)

If files you were editing have changed, Git does its best to merge the
remote changes into your local changes.
It may leave some files to merge by hand.

For example, suppose you have edited `sin.go` but
someone else has committed an independent change.
When you run `git sync`,
you will get the (scary-looking) output:

```
$ git sync
Failed to merge in the changes.
Patch failed at 0023 math: improved Sin, Cos and Tan precision for very large arguments
The copy of the patch that failed is found in:
   /home/you/repo/.git/rebase-apply/patch

When you have resolved this problem, run "git rebase --continue".
If you prefer to skip this patch, run "git rebase --skip" instead.
To check out the original branch and stop rebasing, run "git rebase --abort".
```


If this happens, run

```
$ git status
```

to see which files failed to merge.
The output will look something like this:

```
rebase in progress; onto a24c3eb
You are currently rebasing branch 'mcgillicutty' on 'a24c3eb'.
  (fix conflicts and then run "git rebase --continue")
  (use "git rebase --skip" to skip this patch)
  (use "git rebase --abort" to check out the original branch)

Unmerged paths:
  (use "git reset HEAD <file>..." to unstage)
  (use "git add <file>..." to mark resolution)

	_both modified:   sin.go_
```


The only important part in that transcript is the italicized "both modified"
line: Git failed to merge your changes with the conflicting change.
When this happens, Git leaves both sets of edits in the file,
with conflicts marked by `<<<<<<<` and
`>>>>>>>`.
It is now your job to edit the file to combine them.
Continuing the example, searching for those strings in `sin.go`
might turn up:

```
	arg = scale(arg)
<<<<<<< HEAD
	if arg > 1e9 {
=======
	if arg > 1e10 {
>>>>>>> mcgillicutty
		largeReduce(arg)
```

Git doesn't show it, but suppose the original text that both edits
started with was 1e8; you changed it to 1e10 and the other change to 1e9,
so the correct answer might now be 1e10.  First, edit the section
to remove the markers and leave the correct code:

```
	arg = scale(arg)
	if arg > 1e10 {
		largeReduce(arg)
```

Then tell Git that the conflict is resolved by running

```
$ git add sin.go
```

If you had been editing the file, say for debugging, but do not
care to preserve your changes, you can run
`git reset HEAD sin.go`
to abandon your changes.
Then run `git rebase --continue` to
restore the change commit.

### Reviewing code by others

You can import a change proposed by someone else into your local Git repository.
On the Gerrit review page, click the "Download ▼" link in the upper right
corner, copy the "Checkout" command and run it from your local Git repo.
It should look something like this:

```
$ git fetch https://code.googlesource.com/review refs/changes/21/1221/1 && git checkout FETCH_HEAD
```

To revert, change back to the branch you were working in.

### Submit the change after the review

After the code has been `LGTM`'ed, an approver may
submit it to the master branch using the Gerrit UI.
There is a "Submit" button on the web page for the change
that appears once the change is approved (marked +2).

This checks the change into the repository.
The change description will include a link to the code review,
and the code review will be updated with a link to the change
in the repository.
Since the method used to integrate the changes is "Cherry Pick",
the commit hashes in the repository will be changed by
the submit operation.

### More information

In addition to the information here, the Go community maintains a [CodeReview](https://golang.org/wiki/CodeReview) wiki page.
Feel free to contribute to this page as you learn the review process.

## Contributors

Files in the google-api-go-client repository don't list author names,
both to avoid clutter and to avoid having to keep the lists up to date.
Instead, please add your name to the [`CONTRIBUTORS`](/CONTRIBUTORS)
file as your first code review, keeping the names in sorted order.
