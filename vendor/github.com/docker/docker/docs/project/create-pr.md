<!--[metadata]>
+++
title = "Create a pull request (PR)"
description = "Basic workflow for Docker contributions"
keywords = ["contribute, pull request, review, workflow, beginner, squash,  commit"]
[menu.main]
parent = "smn_contribute"
weight=4
+++
<![end-metadata]-->

# Create a pull request (PR)

A pull request (PR) sends your changes to the Docker maintainers for review. You
create a pull request on GitHub. A pull request "pulls" changes from your forked
repository into the `docker/docker` repository.

You can see <a href="https://github.com/docker/docker/pulls" target="_blank">the
list of active pull requests to Docker</a> on GitHub.

## Check your work

Before you create a pull request, check your work.

1. In a terminal window, go to the root of your `docker-fork` repository. 

        $ cd ~/repos/docker-fork

2. Checkout your feature branch.

        $ git checkout 11038-fix-rhel-link
        Switched to branch '11038-fix-rhel-link'

3. Run the full test suite on your branch.

		$ make test

	All the tests should pass. If they don't, find out why and correct the
	situation. 
    
4. Optionally, if modified the documentation, build the documentation:

		$ make docs

5. Commit and push any changes that result from your checks.

## Rebase your branch

Always rebase and squash your commits before making a pull request. 

1. Checkout your feature branch in your local `docker-fork` repository.

    This is the branch associated with your request.

2. Fetch any last minute changes from `docker/docker`.

        $ git fetch upstream master
        From github.com:docker/docker
         * branch            master     -> FETCH_HEAD

3. Start an interactive rebase.

        $ git rebase -i upstream/master

4. Rebase opens an editor with a list of commits.

        pick 1a79f55 Tweak some of the other text for grammar
        pick 53e4983 Fix a link
        pick 3ce07bb Add a new line about RHEL

5. Replace the `pick` keyword with `squash` on all but the first commit.

        pick 1a79f55 Tweak some of the other text for grammar
        squash 53e4983 Fix a link
        squash 3ce07bb Add a new line about RHEL

    After you save the changes and quit from the editor, git starts
    the rebase, reporting the progress along the way. Sometimes
    your changes can conflict with the work of others. If git
    encounters a conflict, it stops the rebase, and prints guidance
    for how to correct the conflict.

6. Edit and save your commit message.

        $ git commit -s

    Make sure your message includes <a href="../set-up-git" target="_blank">your signature</a>.

7. Force push any changes to your fork on GitHub.

        $ git push -f origin 11038-fix-rhel-link
        
## Create a PR on GitHub

You create and manage PRs on GitHub:

1. Open your browser to your fork on GitHub.

    You should see the latest activity from your branch.

    ![Latest commits](/project/images/latest_commits.png)


2. Click "Compare & pull request."

    The system displays the pull request dialog. 

    ![PR dialog](/project/images/to_from_pr.png)

    The pull request compares your changes to the `master` branch on the
    `docker/docker` repository.

3. Edit the dialog's description and add a reference to the issue you are fixing.

    GitHub helps you out by searching for the issue as you type.

    ![Fixes issue](/project/images/fixes_num.png)

4. Scroll down and verify the PR contains the commits and changes you expect.

    For example, is the file count correct? Are the changes in the files what
    you expect?

    ![Commits](/project/images/commits_expected.png)

5. Press "Create pull request".

    The system creates the request and opens it for you in the `docker/docker`
    repository.

    ![Pull request made](/project/images/pull_request_made.png)


## Where to go next

Congratulations, you've created your first pull request to Docker. The next
step is for you learn how to [participate in your PR's
review](/project/review-pr/).
