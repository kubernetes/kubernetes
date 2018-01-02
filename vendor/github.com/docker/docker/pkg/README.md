pkg/ is a collection of utility packages used by the Docker project without being specific to its internals.

Utility packages are kept separate from the docker core codebase to keep it as small and concise as possible.
If some utilities grow larger and their APIs stabilize, they may be moved to their own repository under the
Docker organization, to facilitate re-use by other projects. However that is not the priority.

The directory `pkg` is named after the same directory in the camlistore project. Since Brad is a core
Go maintainer, we thought it made sense to copy his methods for organizing Go code :) Thanks Brad!

Because utility packages are small and neatly separated from the rest of the codebase, they are a good
place to start for aspiring maintainers and contributors. Get in touch if you want to help maintain them!
