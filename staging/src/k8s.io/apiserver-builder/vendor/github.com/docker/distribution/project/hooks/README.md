Git Hooks
=========

To enforce valid and properly-formatted code, there is CI in place which runs `gofmt`, `golint`, and `go vet` against code in the repository.

As an aid to prevent committing invalid code in the first place, a git pre-commit hook has been added to the repository, found in [pre-commit](./pre-commit). As it is impossible to automatically add linked hooks to a git repository, this hook should be linked into your `.git/hooks/pre-commit`, which can be done by running the `configure-hooks.sh` script in this directory. This script is the preferred method of configuring hooks, as it will be updated as more are added.