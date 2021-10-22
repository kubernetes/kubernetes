Step 2: Create an Issue
========================

Every patch / Pull Request requires a corresponding issue. If you're fixing
a bug for an existing issue, then there's no need to create a new issue.

However, if no prior issue exists, you must create an issue.

Reporting a Bug
---------------

When reporting a bug, please try to provide as much information as you
can.

The following issues are good examples for reporting a bug:

* https://github.com/gophercloud/gophercloud/issues/108
* https://github.com/gophercloud/gophercloud/issues/212
* https://github.com/gophercloud/gophercloud/issues/424
* https://github.com/gophercloud/gophercloud/issues/588
* https://github.com/gophercloud/gophercloud/issues/629
* https://github.com/gophercloud/gophercloud/issues/647

Feature Request
---------------

If you've noticed that a feature is missing from Gophercloud, you'll also
need to create an issue before doing any work. This is start a discussion about
whether or not the feature should be included in Gophercloud. We don't want to
want to see you put in hours of work only to learn that the feature is out of
scope of the project.

Feature requests can come in different forms:

### Adding a Feature to Gophercloud Core

The "core" of Gophercloud is the code which supports API requests and
responses: pagination, error handling, building request bodies, and parsing
response bodies are all examples of core code.

Modifications to core will usually have the most amount of discussion than
other requests since a change to core will affect _all_ of Gophercloud.

The following issues are examples of core change discussions:

* https://github.com/gophercloud/gophercloud/issues/310
* https://github.com/gophercloud/gophercloud/issues/613
* https://github.com/gophercloud/gophercloud/issues/729
* https://github.com/gophercloud/gophercloud/issues/713

### Adding a Missing Field

If you've found a missing field in an existing struct, submit an issue to
request having it added. These kinds of issues are pretty easy to report
and resolve.

You should also provide a link to the actual service's Python code which
defines the missing field.

The following issues are examples of missing fields:

* https://github.com/gophercloud/gophercloud/issues/620
* https://github.com/gophercloud/gophercloud/issues/621
* https://github.com/gophercloud/gophercloud/issues/658

There's one situation which can make adding fields more difficult: if the field
is part of an API extension rather than the base API itself. An example of this
can be seen in [this](https://github.com/gophercloud/gophercloud/issues/749)
issue.

Here, a user reported fields missing in the `Get` function of
`networking/v2/networks`. The fields reported missing weren't missing at all,
they're just part of various Networking extensions located in
`networking/v2/extensions`.

### Adding a Missing API Call

If you've found a missing API action, create an issue with details of
the action. For example:

* https://github.com/gophercloud/gophercloud/issues/715
* https://github.com/gophercloud/gophercloud/issues/719

You'll want to make sure the API call is part of the upstream OpenStack project
and not an extension created by a third-party or vendor. Gophercloud only
supports the OpenStack projects proper.

### Adding a Missing API Suite

Adding support to a missing suite of API calls will require more than one Pull
Request. However, you can use a single issue for all PRs.

Examples of issues which track the addition of a missing API suite are:

* https://github.com/gophercloud/gophercloud/issues/539
* https://github.com/gophercloud/gophercloud/issues/555
* https://github.com/gophercloud/gophercloud/issues/571
* https://github.com/gophercloud/gophercloud/issues/583
* https://github.com/gophercloud/gophercloud/issues/605

Note how the issue breaks down the implementation by request types (Create,
Update, Delete, Get, List).

Also note how these issues provide links to the service's Python code. These
links are not required for _issues_, but it's usually a good idea to provide
them, anyway. These links _are required_ for PRs and that will be covered in
detail in a later step of this tutorial.

### Adding a Missing OpenStack Project

These kinds of feature additions are large undertakings. Adding support for
an entire OpenStack project is something the Gophercloud team very much
appreciates, but you should be prepared for several weeks of work and
interaction with the Gophercloud team.

An example of how to create an issue for an entire project can be seen
here:

* https://github.com/gophercloud/gophercloud/issues/723

---

With all of the above in mind, proceed to [Step 3](step-03-code-hunting.md) to
learn about Code Hunting.
