# Contributing to Testify

So you'd like to contribute to Testify? First of all, thank you! Testify is widely used, so each
contribution has a significant impact within the Golang community! Below you'll find everything you
need to know to get up to speed on the project.

## Philosophy

The Testify maintainers generally attempt to follow widely accepted practices within the Golang
community. That being said, the first priority is always to make sure that the package is useful to
the community. A few general guidelines are listed here:

*Keep it simple (whenever practical)* - Try not to expand the API unless the new surface area
provides meaningful benefits. For example, don't add functions because they might be useful to
someone, someday. Add what is useful to specific users, today.

*Ease of use is paramount* - This means good documentation and package organization. It also means
that we should try hard to use meaningful, descriptive function names, avoid breaking the API
unnecessarily, and try not to surprise the user.

*Quality isn't an afterthought* - Testify is a testing library, so it seems reasonable that we
should have a decent test suite. This is doubly important because a bug in Testify doesn't just mean
a bug in our users' code, it means a bug in our users' tests, which means a potentially unnoticed
and hard-to-find bug in our users' code.

## Pull Requests

We welcome pull requests! Please include the following in the description:

  * Motivation, why your change is important or helpful
  * Example usage (if applicable)
  * Whether you intend to add / change behavior or fix a bug

Please be aware that the maintainers may ask for changes. This isn't a commentary on the quality of
your idea or your code. Testify is the result of many contributions from many individuals, so we
need to enforce certain practices and patterns to keep the package easy for others to understand.
Essentially, we recognize that there are often many good ways to do a given thing, but we have to
pick one and stick with it.

See `MAINTAINERS.md` for a list of users who can approve / merge your changes.

## Issues

If you find a bug or think of a useful feature you'd like to see added to Testify, the best thing
you can do is make the necessary changes and open a pull request (see above). If that isn't an
option, or if you'd like to discuss your change before you write the code, open an issue!

Please provide enough context in the issue description that other members of the community can
easily understand what it is that you'd like to see.

