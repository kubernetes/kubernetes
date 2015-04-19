# How to contribute

We're always happy about useful new pull requests. Keep in mind that the better
your pull request is, the easier it can be added to `xml2js`. As such please
make sure your patch is ok:

 * `xml2js` is written in CoffeeScript. Please don't send patches to
   the JavaScript source, as it get's overwritten by the CoffeeScript
   compiler. The reason we have the JS code in the repository is for easier
   use with eg. `git submodule`
 * Make sure that the unit tests still all pass. Failing unit tests mean that
   someone *will* run into a bug, if we accept your pull request.
 * Please, add a unit test with your pull request, to show what was broken and
   is now fixed or what was impossible and now works due to your new code.
 * If you add a new feature, please add some documentation that it exists.

If you like, you can add yourself in the `package.json` as contributor if you
deem your contribution significant enough. Otherwise, we will decide and maybe
add you.
