<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# On Collaborative Development

Kubernetes is open source, but many of the people working on it do so as their
day job. In order to avoid forcing people to be "at work" effectively 24/7, we
want to establish some semi-formal protocols around development. Hopefully these
rules make things go more smoothly. If you find that this is not the case,
please complain loudly.

## Patches welcome

First and foremost: as a potential contributor, your changes and ideas are
welcome at any hour of the day or night, weekdays, weekends, and holidays.
Please do not ever hesitate to ask a question or send a PR.

## Code reviews

All changes must be code reviewed. For non-maintainers this is obvious, since
you can't commit anyway. But even for maintainers, we want all changes to get at
least one review, preferably (for non-trivial changes obligatorily) from someone
who knows the areas the change touches. For non-trivial changes we may want two
reviewers. The primary reviewer will make this decision and nominate a second
reviewer, if needed. Except for trivial changes, PRs should not be committed
until relevant parties (e.g. owners of the subsystem affected by the PR) have
had a reasonable chance to look at PR in their local business hours.

Most PRs will find reviewers organically. If a maintainer intends to be the
primary reviewer of a PR they should set themselves as the assignee on GitHub
and say so in a reply to the PR. Only the primary reviewer of a change should
actually do the merge, except in rare cases (e.g. they are unavailable in a
reasonable timeframe).

If a PR has gone 2 work days without an owner emerging, please poke the PR
thread and ask for a reviewer to be assigned.

Except for rare cases, such as trivial changes (e.g. typos, comments) or
emergencies (e.g. broken builds), maintainers should not merge their own
changes.

Expect reviewers to request that you avoid [common go style
mistakes](https://github.com/golang/go/wiki/CodeReviewComments) in your PRs.

## Assigned reviews

Maintainers can assign reviews to other maintainers, when appropriate. The
assignee becomes the shepherd for that PR and is responsible for merging the PR
once they are satisfied with it or else closing it. The assignee might request
reviews from non-maintainers.

## Merge hours

Maintainers will do merges of appropriately reviewed-and-approved changes during
their local "business hours" (typically 7:00 am Monday to 5:00 pm (17:00h)
Friday). PRs that arrive over the weekend or on holidays will only be merged if
there is a very good reason for it and if the code review requirements have been
met. Concretely this means that nobody should merge changes immediately before
going to bed for the night.

There may be discussion an even approvals granted outside of the above hours,
but merges will generally be deferred.

If a PR is considered complex or controversial, the merge of that PR should be
delayed to give all interested parties in all timezones the opportunity to
provide feedback. Concretely, this means that such PRs should be held for 24
hours before merging. Of course "complex" and "controversial" are left to the
judgment of the people involved, but we trust that part of being a committer is
the judgment required to evaluate such things honestly, and not be motivated by
your desire (or your cube-mate's desire) to get their code merged. Also see
"Holds" below, any reviewer can issue a "hold" to indicate that the PR is in
fact complicated or complex and deserves further review.

PRs that are incorrectly judged to be merge-able, may be reverted and subject to
re-review, if subsequent reviewers believe that they in fact are controversial
or complex.


## Holds

Any maintainer or core contributor who wants to review a PR but does not have
time immediately may put a hold on a PR simply by saying so on the PR discussion
and offering an ETA measured in single-digit days at most. Any PR that has a
hold shall not be merged until the person who requested the hold acks the
review, withdraws their hold, or is overruled by a preponderance of maintainers.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/collab.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
