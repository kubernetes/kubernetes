# How to Contribute

CSI is under [Apache 2.0](LICENSE) and accepts contributions via GitHub pull requests.
This document outlines some of the conventions on development workflow, commit message formatting, contact points and other resources to make it easier to get your contribution accepted.

## Markdown style

To keep consistency throughout the Markdown files in the CSI spec, all files should be formatted one sentence per line.
This fixes two things: it makes diffing easier with git and it resolves fights about line wrapping length.
For example, this paragraph will span three lines in the Markdown source.

## Code style

This also applies to the code snippets in the markdown files.

* Please wrap the code at 72 characters.

## Comments

This also applies to the code snippets in the markdown files.

* End each sentence within a comment with a punctuation mark (please note that we generally prefer periods); this applies to incomplete sentences as well.
* For trailing comments, leave one space between the end of the code and the beginning of the comment.

## Git commit

Prior to committing code please run `make` in order to update the protobuf file and any language bindings.

### Commit Style

Each commit should represent a single logical (atomic) change: this makes your changes easier to review.

* Try to avoid unrelated cleanups (e.g., typo fixes or style nits) in the same commit that makes functional changes.
  While typo fixes are great, including them in the same commit as functional changes makes the commit history harder to read.
* Developers often make incremental commits to save their progress when working on a change, and then “rewrite history” (e.g., using `git rebase -i`) to create a clean set of commits once the change is ready to be reviewed.

Simple house-keeping for clean git history.
Read more on [How to Write a Git Commit Message](http://chris.beams.io/posts/git-commit/) or the Discussion section of [`git-commit(1)`](http://git-scm.com/docs/git-commit).

* Separate the subject from body with a blank line.
* Limit the subject line to 50 characters.
* Capitalize the subject line.
* Do not end the subject line with a period.
* Use the imperative mood in the subject line.
* Wrap the body at 72 characters.
* Use the body to explain what and why vs. how.
  * If there was important/useful/essential conversation or information, copy or include a reference.
* When possible, one keyword to scope the change in the subject (i.e. "README: ...", "tool: ...").
