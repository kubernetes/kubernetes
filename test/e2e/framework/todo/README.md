This directory holds sub packages which, in contrast to other sub packages
under test/e2e/framework, may use test/e2e/framework because that is not
depending on them.

This is an interim solution for moving code without causing cycling
dependencies. All code will be moved from here into the normal sub packages
when the refactoring is done.
