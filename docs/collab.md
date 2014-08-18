# On Collaborative Development

Kubernetes is open source, but many of the people working on it do so as their day job.  In order to avoid forcing people to be "at work" effectively 24/7, we want to establish some semi-formal protocols around development.  Hopefully these rules make things go more smoothly.  If you find that this is not the case, please complain loudly.

## Patches welcome

First and foremost: as a potential contributor, your changes and ideas are welcome at any hour of the day or night, weekdays, weekends, and holidays.  Please do not ever hesitate to ask a question or send a PR.

## Timezones and calendars

For the time being, most of the people working on this project are in the US and on Pacific time.  Any times mentioned henceforth will refer to this timezone.  Any references to "work days" will refer to the US calendar.

## Code reviews

All changes must be code reviewed.  For non-maintainers this is obvious, since you can't commit anyway.  But even for maintainers, we want all changes to get at least one review, preferably from someone who knows the areas the change touches.  For non-trivial changes we may want two reviewers.  The primary reviewer will make this decision and nominate a second reviewer, if needed.  Except for trivial changes, PRs should sit for at least a 2 hours to allow for wider review.

Most PRs will find reviewers organically.  If a maintainer intends to be the primary reviewer of a PR they should set themselves as the assignee on GitHub and say so in a reply to the PR.  Only the primary reviewer of a change should actually do the merge, except in rare cases (e.g. they are unavailable in a reasonable timeframe).

If a PR has gone 2 work days without an owner emerging, please poke the PR thread and ask for a reviewer to be assigned.

Except for rare cases, such as trivial changes (e.g. typos, comments) or emergencies (e.g. broken builds), maintainers should not merge their own changes.

Expect reviewers to request that you avoid [common go style mistakes](https://code.google.com/p/go-wiki/wiki/CodeReviewComments) in your PRs.

## Merge hours

Maintainers will do merges between the hours of 7:00 am Monday and 7:00 pm (19:00h) Friday.  PRs that arrive over the weekend or on holidays will only be merged if there is a very good reason for it and if the code review requirements have been met.

There may be discussion an even approvals granted outside of the above hours, but merges will generally be deferred.

## Holds

Any maintainer or core contributor who wants to review a PR but does not have time immediately may put a hold on a PR simply by saying so on the PR discussion and offering an ETA measured in single-digit days at most.  Any PR that has a hold shall not be merged until the person who requested the hold acks the review, withdraws their hold, or is overruled by a preponderance of maintainers.
