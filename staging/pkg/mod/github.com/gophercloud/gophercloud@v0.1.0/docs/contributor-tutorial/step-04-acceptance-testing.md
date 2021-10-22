Step 4: Acceptance Testing
==========================

If we haven't started working on the feature or bug fix, why are we talking
about Acceptance Testing now?

Before you implement a feature or bug fix, you _must_ be able to test your code
in a working OpenStack environment. Please do not submit code which you have
only tested with offline unit tests.

Blindly submitting code is dangerous to the Gophercloud project. Developers
from all over the world use Gophercloud in many different projects. If you
submit code which is untested, it can cause these projects to break or become
unstable.

And, to be frank, submitting untested code will inevitably cause someone else
to have to spend time fixing it.

If you don't have an OpenStack environment to test with, we have lots of
documentation [here](/acceptance) to help you build your own small OpenStack
environment for testing.

---

Once you've confirmed you are able to test your code, proceed to
[Step 5](step-05-pull-requests.md) to (finally!) start working on a Pull
Request.
