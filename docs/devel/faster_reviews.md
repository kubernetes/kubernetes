# How to get faster PR reviews

Most of what is written here is not at all specific to Kubernetes, but it bears
being written down in the hope that it will occasionally remind people of "best
practices" around code reviews.

You've just had a brilliant idea on how to make Kubernetes better. Let's call
that idea "Feature-X". Feature-X is not even that complicated. You have a pretty
good idea of how to implement it. You jump in and implement it, fixing a bunch
of stuff along the way. You send your PR - this is awesome! And it sits. And
sits. A week goes by and nobody reviews it. Finally someone offers a few
comments, which you fix up and wait for more review. And you wait. Another
week or two goes by. This is horrible.

What went wrong? One particular problem that comes up frequently is this - your
PR is too big to review. You've touched 39 files and have 8657 insertions. When
your would-be reviewers pull up the diffs they run away - this PR is going to
take 4 hours to review and they don't have 4 hours right now. They'll get to it
later, just as soon as they have more free time (ha!).

Let's talk about how to avoid this.

## 0. Familiarize yourself with project conventions

* [Development guide](development.md)
* [Coding conventions](coding-conventions.md)
* [API conventions](api-conventions.md)
* [Kubectl conventions](kubectl-conventions.md)

## 1. Don't build a cathedral in one PR

Are you sure Feature-X is something the Kubernetes team wants or will accept, or
that it is implemented to fit with other changes in flight? Are you willing to
bet a few days or weeks of work on it? If you have any doubt at all about the
usefulness of your feature or the design - make a proposal doc (in
docs/proposals; for example [the QoS proposal](http://prs.k8s.io/11713)) or a
sketch PR (e.g., just the API or Go interface) or both. Write or code up just
enough to express the idea and the design and why you made those choices, then
get feedback on this. Be clear about what type of feedback you are asking for.
Now, if we ask you to change a bunch of facets of the design, you won't have to
re-write it all.

## 2. Smaller diffs are exponentially better

Small PRs get reviewed faster and are more likely to be correct than big ones.
Let's face it - attention wanes over time. If your PR takes 60 minutes to
review, I almost guarantee that the reviewer's eye for detail is not as keen in
the last 30 minutes as it was in the first. This leads to multiple rounds of
review when one might have sufficed. In some cases the review is delayed in its
entirety by the need for a large contiguous block of time to sit and read your
code.

Whenever possible, break up your PRs into multiple commits. Making a series of
discrete commits is a powerful way to express the evolution of an idea or the
different ideas that make up a single feature. There's a balance to be struck,
obviously. If your commits are too small they become more cumbersome to deal
with. Strive to group logically distinct ideas into separate commits.

For example, if you found that Feature-X needed some "prefactoring" to fit in,
make a commit that JUST does that prefactoring. Then make a new commit for
Feature-X. Don't lump unrelated things together just because you didn't think
about prefactoring. If you need to, fork a new branch, do the prefactoring
there and send a PR for that. If you can explain why you are doing seemingly
no-op work ("it makes the Feature-X change easier, I promise") we'll probably be
OK with it.

Obviously, a PR with 25 commits is still very cumbersome to review, so use
common sense.

## 3. Multiple small PRs are often better than multiple commits

If you can extract whole ideas from your PR and send those as PRs of their own,
you can avoid the painful problem of continually rebasing. Kubernetes is a
fast-moving codebase - lock in your changes ASAP, and make merges be someone
else's problem.

Obviously, we want every PR to be useful on its own, so you'll have to use
common sense in deciding what can be a PR vs. what should be a commit in a larger
PR. Rule of thumb - if this commit or set of commits is directly related to
Feature-X and nothing else, it should probably be part of the Feature-X PR. If
you can plausibly imagine someone finding value in this commit outside of
Feature-X, try it as a PR.

Don't worry about flooding us with PRs. We'd rather have 100 small, obvious PRs
than 10 unreviewable monoliths.

## 4. Don't rename, reformat, comment, etc in the same PR

Often, as you are implementing Feature-X, you find things that are just wrong.
Bad comments, poorly named functions, bad structure, weak type-safety. You
should absolutely fix those things (or at least file issues, please) - but not
in this PR. See the above points - break unrelated changes out into different
PRs or commits. Otherwise your diff will have WAY too many changes, and your
reviewer won't see the forest because of all the trees.

## 5. Comments matter

Read up on GoDoc - follow those general rules. If you're writing code and you
think there is any possible chance that someone might not understand why you did
something (or that you won't remember what you yourself did), comment it. If
you think there's something pretty obvious that we could follow up on, add a
TODO. Many code-review comments are about this exact issue.

## 5. Tests are almost always required

Nothing is more frustrating than doing a review, only to find that the tests are
inadequate or even entirely absent. Very few PRs can touch code and NOT touch
tests. If you don't know how to test Feature-X - ask!  We'll be happy to help
you design things for easy testing or to suggest appropriate test cases.

## 6. Look for opportunities to generify

If you find yourself writing something that touches a lot of modules, think hard
about the dependencies you are introducing between packages. Can some of what
you're doing be made more generic and moved up and out of the Feature-X package?
Do you need to use a function or type from an otherwise unrelated package? If
so, promote! We have places specifically for hosting more generic code.

Likewise if Feature-X is similar in form to Feature-W which was checked in last
month and it happens to exactly duplicate some tricky stuff from Feature-W,
consider prefactoring core logic out and using it in both Feature-W and
Feature-X. But do that in a different commit or PR, please.

## 7. Fix feedback in a new commit

Your reviewer has finally sent you some feedback on Feature-X. You make a bunch
of changes and ... what?  You could patch those into your commits with git
"squash" or "fixup" logic.  But that makes your changes hard to verify. Unless
your whole PR is pretty trivial, you should instead put your fixups into a new
commit and re-push. Your reviewer can then look at that commit on its own - so
much faster to review than starting over.

We might still ask you to clean up your commits at the very end, for the sake
of a more readable history, but don't do this until asked, typically at the
point where the PR would otherwise be tagged LGTM.

General squashing guidelines:

* Sausage => squash

  When there are several commits to fix bugs in the original commit(s), address
reviewer feedback, etc. Really we only want to see the end state and commit
message for the whole PR.

* Layers => don't squash

  When there are independent changes layered upon each other to achieve a single
goal. For instance, writing a code munger could be one commit, applying it could
be another, and adding a precommit check could be a third. One could argue they
should be separate PRs, but there's really no way to test/review the munger
without seeing it applied, and there needs to be a precommit check to ensure the
munged output doesn't immediately get out of date.

A commit, as much as possible, should be a single logical change. Each commit
should always have a good title line (<70 characters) and include an additional
description paragraph describing in more detail the change intended. Do not link
pull requests by `#` in a commit description, because GitHub creates lots of
spam. Instead, reference other PRs via the PR your commit is in.

## 8. KISS, YAGNI, MVP, etc

Sometimes we need to remind each other of core tenets of software design - Keep
It Simple, You Aren't Gonna Need It, Minimum Viable Product, and so on. Adding
features "because we might need it later" is antithetical to software that
ships. Add the things you need NOW and (ideally) leave room for things you
might need later - but don't implement them now.

## 9. Push back

We understand that it is hard to imagine, but sometimes we make mistakes. It's
OK to push back on changes requested during a review. If you have a good reason
for doing something a certain way, you are absolutely allowed to debate the
merits of a requested change. You might be overruled, but you might also
prevail. We're mostly pretty reasonable people. Mostly.

## 10. I'm still getting stalled - help?!

So, you've done all that and you still aren't getting any PR love? Here's some
things you can do that might help kick a stalled process along:

   * Make sure that your PR has an assigned reviewer (assignee in GitHub). If
this is not the case, reply to the PR comment stream asking for one to be
assigned.

   * Ping the assignee (@username) on the PR comment stream asking for an
estimate of when they can get to it.

   * Ping the assignee by email (many of us have email addresses that are well
published or are the same as our GitHub handle @google.com or @redhat.com).

   * Ping the [team](https://github.com/orgs/kubernetes/teams) (via @team-name)
that works in the area you're submitting code.

If you think you have fixed all the issues in a round of review, and you haven't
heard back, you should ping the reviewer (assignee) on the comment stream with a
"please take another look" (PTAL) or similar comment indicating you are done and
you think it is ready for re-review. In fact, this is probably a good habit for
all PRs.

One phenomenon of open-source projects (where anyone can comment on any issue)
is the dog-pile - your PR gets so many comments from so many people it becomes
hard to follow. In this situation you can ask the primary reviewer (assignee)
whether they want you to fork a new PR to clear out all the comments. Remember:
you don't HAVE to fix every issue raised by every person who feels like
commenting, but you should at least answer reasonable comments with an
explanation.

## Final: Use common sense

Obviously, none of these points are hard rules. There is no document that can
take the place of common sense and good taste. Use your best judgment, but put
a bit of thought into how your work can be made easier to review. If you do
these things your PRs will flow much more easily.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/faster_reviews.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
