<!--[metadata]>
+++
title = "Advanced contributing"
description = "Explains workflows for refactor and design proposals"
keywords = ["contribute, project, design, refactor,  proposal"]
[menu.main]
parent = "smn_contribute"
weight=6
+++
<![end-metadata]-->

# Advanced contributing

In this section, you learn about the more advanced contributions you can make.
They are advanced because they have a more involved workflow or require greater
programming experience. Don't be scared off though, if you like to stretch and
challenge yourself, this is the place for you.

This section gives generalized instructions for advanced contributions. You'll
read about the workflow but there are not specific descriptions of commands.
Your goal should be to understand the processes described.

At this point, you should have read and worked through the earlier parts of
the project contributor guide. You should also have
<a href="../make-a-contribution/" target="_blank"> made at least one project contribution</a>.

## Refactor or cleanup proposal

A refactor or cleanup proposal changes Docker's internal structure without
altering the external behavior. To make this type of proposal:

1. Fork `docker/docker`.

2. Make your changes in a feature branch.

3. Sync and rebase with `master` as you work.

3. Run the full test suite.

4. Submit your code through a pull request (PR).

    The PR's title should have the format:

    **Cleanup:** _short title_

    If your changes required logic changes, note that in your request.
	
5. Work through Docker's review process until merge.


## Design proposal

A design proposal solves a problem or adds a feature to the Docker software.
The process for submitting design proposals requires two pull requests, one
for the design and one for the implementation.

![Simple process](/project/images/proposal.png)

The important thing to notice is that both the design pull request and the
implementation pull request go through a review. In other words, there is
considerable time commitment in a design proposal; so, you might want to pair
with someone on design work.

The following provides greater detail on the process:

1. Come up with an idea.

    Ideas usually come from limitations users feel working with a product. So,
    take some time to really use Docker. Try it on different platforms; explore
    how it works with different web applications. Go to some community events
    and find out what other users want.

2. Review existing issues and proposals to make sure no other user is proposing a similar idea.

    The design proposals are <a
    href="https://github.com/docker/docker/pulls?q=is%3Aopen+is%3Apr+label%
    3Akind%2Fproposal" target="_blank">all online in our GitHub pull requests</a>. 
    
3. Talk to the community about your idea.

    We have lots of <a href="../get-help/" target="_blank">community forums</a>
    where you can get feedback on your idea. Float your idea in a forum or two
    to get some commentary going on it.

4. Fork `docker/docker` and clone the repo to your local host.

5. Create a new Markdown file in the area you wish to change.  

    For example, if you want to redesign our daemon create a new file under the
    `daemon/` folder. 

6. Name the file descriptively, for example `redesign-daemon-proposal.md`.

7. Write a proposal for your change into the file.

    This is a Markdown file that describes your idea. Your proposal
    should include information like:

    * Why is this change needed or what are the use cases?
    * What are the requirements this change should meet?
    * What are some ways to design/implement this feature?
    * Which design/implementation do you think is best and why?
    * What are the risks or limitations of your proposal?

    This is your chance to convince people your idea is sound. 

8. Submit your proposal in a pull request to `docker/docker`.

    The title should have the format:

    **Proposal:** _short title_

    The body of the pull request should include a brief summary of your change
    and then say something like "_See the file for a complete description_".

9. Refine your proposal through review.

    The maintainers and the community review your proposal. You'll need to
    answer questions and sometimes explain or defend your approach. This is
    chance for everyone to both teach and learn.

10. Pull request accepted.

    Your request may also be rejected. Not every idea is a good fit for Docker.
    Let's assume though your proposal succeeded. 

11. Implement your idea.

    Implementation uses all the standard practices of any contribution.

    * fork `docker/docker`
    * create a feature branch
    * sync frequently back to master
    * test as you go and full test before a PR

    If you run into issues, the community is there to help.

12. When you have a complete implementation, submit a pull request back to `docker/docker`.

13. Review and iterate on your code.

    If you are making a large code change, you can expect greater scrutiny
    during this phase. 

14. Acceptance and merge!

## About the advanced process

Docker is a large project. Our core team gets a great many design proposals.
Design proposal discussions can span days, weeks, and longer. The number of comments can reach the 100s.
In that situation, following the discussion flow and the decisions reached is crucial.

Making a pull request with a design proposal simplifies this process:
* you can leave comments on specific design proposal line
* replies around line are easy to track
* as a proposal changes and is updated, pages reset as line items resolve
* GitHub maintains the entire history

While proposals in pull requests do not end up merged into a master repository, they provide a convenient tool for managing the design process.
