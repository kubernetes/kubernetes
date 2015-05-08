# Contributing

**Important:** these GitHub issues are for *bug reports and feature requests only*. Please use [StackOverflow](http://stackoverflow.com/questions/tagged/d3.js) or the [d3-js Google group](https://groups.google.com/d/forum/d3-js) for general help.

If you’re looking for ways to contribute, please [peruse open issues](https://github.com/mbostock/d3/issues?milestone=&page=1&state=open). The icebox is a good place to find ideas that are not currently in development. If you already have an idea, please check past issues to see whether your idea or a similar one was previously discussed.

Before submitting a pull request, consider implementing a live example first, say using [bl.ocks.org](http://bl.ocks.org). Real-world use cases go a long way to demonstrating the usefulness of a proposed feature. The more complex a feature’s implementation, the more usefulness it should provide. Share your demo using the #d3js tag on Twitter or by sending it to the [d3-js Google group](https://groups.google.com/d/forum/d3-js).

If your proposed feature does not involve changing core functionality, consider submitting it instead as a [D3 plugin](https://github.com/d3/d3-plugins). New core features should be for general use, whereas plugins are suitable for more specialized use cases. When in doubt, it’s easier to start with a plugin before “graduating” to core.

To contribute new documentation or add examples to the gallery, just [edit the Wiki](https://github.com/mbostock/d3/wiki)!

## How to Submit a Pull Request

1. Click the “Fork” button to create your personal fork of the D3 repository.

2. After cloning your fork of the D3 repository in the terminal, run `npm install` to install D3’s dependencies.

3. Create a new branch for your new feature. For example: `git checkout -b my-awesome-feature`. A dedicated branch for your pull request means you can develop multiple features at the same time, and ensures that your pull request is stable even if you later decide to develop an unrelated feature.

4. The `d3.js` and `d3.min.js` files are built from source files in the `src` directory. _Do not edit `d3.js` directly._ Instead, edit the source files, and then run `make` to build the generated files.

5. Use `make test` to run tests and verify your changes. If you are adding a new feature, you should add new tests! If you are changing existing functionality, make sure the existing tests run, or update them as appropriate.

6. Sign D3’s [Individual Contributor License Agreement](https://docs.google.com/forms/d/1CzjdBKtDuA8WeuFJinadx956xLQ4Xriv7-oDvXnZMaI/viewform). Unless you are submitting a trivial patch (such as fixing a typo), this form is needed to verify that you are able to contribute.

7. Submit your pull request, and good luck!
