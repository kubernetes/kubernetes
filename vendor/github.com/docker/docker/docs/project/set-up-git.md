<!--[metadata]>
+++
title = "Configure Git for contributing"
description = "Describes how to set up your local machine and repository"
keywords = ["GitHub account, repository, clone, fork, branch, upstream, Git, Go, make "]
[menu.main]
parent = "smn_develop"
weight=4
+++
<![end-metadata]-->

# Configure Git for contributing

Work through this page to configure Git and a repository you'll use throughout
the Contributor Guide. The work you do further in the guide, depends on the work
you do here. 

## Fork and clone the Docker code

Before contributing, you first fork the Docker code repository. A fork copies
a repository at a particular point in time. GitHub tracks for you where a fork
originates.

As you make contributions, you change your fork's code. When you are ready,
you make a pull request back to the original Docker repository. If you aren't
familiar with this workflow, don't worry, this guide walks you through all the
steps. 

To fork and clone Docker:

1. Open a browser and log into GitHub with your account.

2. Go to the <a href="https://github.com/docker/docker"
target="_blank">docker/docker repository</a>.

3. Click the "Fork" button in the upper right corner of the GitHub interface.

    ![Branch Signature](/project/images/fork_docker.png)

    GitHub forks the repository to your GitHub account. The original
    `docker/docker` repository becomes a new fork `YOUR_ACCOUNT/docker` under
    your account.

4. Copy your fork's clone URL from GitHub.

    GitHub allows you to use HTTPS or SSH protocols for clones. You can use the
    `git` command line or clients like Subversion to clone a repository. 

    ![Copy clone URL](/project/images/copy_url.png)

    This guide assume you are using the HTTPS protocol and the `git` command
    line. If you are comfortable with SSH and some other tool, feel free to use
    that instead. You'll need to convert what you see in the guide to what is
    appropriate to your tool.

5. Open a terminal window on your local host and change to your home directory. 

        $ cd ~
        
  In Windows, you'll work in your Boot2Docker window instead of Powershell or
  a `cmd` window.

6. Create a `repos` directory.

        $ mkdir repos

7. Change into your `repos` directory.

        $ cd repos

5. Clone the fork to your local host into a repository called `docker-fork`.

        $ git clone https://github.com/moxiegirl/docker.git docker-fork

    Naming your local repo `docker-fork` should help make these instructions
    easier to follow; experienced coders don't typically change the name.

6. Change directory into your new `docker-fork` directory.

        $ cd docker-fork

    Take a moment to familiarize yourself with the repository's contents. List
    the contents. 

##  Set your signature and an upstream remote

When you contribute to Docker, you must certify you agree with the 
<a href="http://developercertificate.org/" target="_blank">Developer Certificate of Origin</a>.
You indicate your agreement by signing your `git` commits like this:

    Signed-off-by: Pat Smith <pat.smith@email.com>

To create a signature, you configure your username and email address in Git.
You can set these globally or locally on just your `docker-fork` repository.
You must sign with your real name. We don't accept anonymous contributions or
contributions through pseudonyms.

As you change code in your fork, you'll want to keep it in sync with the changes
others make in the `docker/docker` repository. To make syncing easier, you'll
also add a _remote_ called `upstream` that points to `docker/docker`. A remote
is just another project version hosted on the internet or network.

To configure your username, email, and add a remote:

1. Change to the root of your `docker-fork` repository.

        $ cd docker-fork

2. Set your `user.name` for the repository.

        $ git config --local user.name "FirstName LastName"

3. Set your `user.email` for the repository.

        $ git config --local user.email "emailname@mycompany.com"

4. Set your local repo to track changes upstream, on the `docker` repository. 

        $ git remote add upstream https://github.com/docker/docker.git

7. Check the result in your `git` configuration.

        $ git config --local -l
        core.repositoryformatversion=0
        core.filemode=true
        core.bare=false
        core.logallrefupdates=true
        remote.origin.url=https://github.com/moxiegirl/docker.git
        remote.origin.fetch=+refs/heads/*:refs/remotes/origin/*
        branch.master.remote=origin
        branch.master.merge=refs/heads/master
        user.name=Mary Anthony
        user.email=mary@docker.com
        remote.upstream.url=https://github.com/docker/docker.git
        remote.upstream.fetch=+refs/heads/*:refs/remotes/upstream/*

	To list just the remotes use:

        $ git remote -v
        origin	https://github.com/moxiegirl/docker.git (fetch)
        origin	https://github.com/moxiegirl/docker.git (push)
        upstream	https://github.com/docker/docker.git (fetch)
        upstream	https://github.com/docker/docker.git (push)

## Create and push a branch

As you change code in your fork, make your changes on a repository branch.
The branch name should reflect what you are working on. In this section, you
create a branch, make a change, and push it up to your fork. 

This branch is just for testing your config for this guide. The changes are part
of a dry run, so the branch name will be dry-run-test. To create and push
the branch to your fork on GitHub:

1. Open a terminal and go to the root of your `docker-fork`.

        $ cd docker-fork

2. Create a `dry-run-test` branch.

        $ git checkout -b dry-run-test

    This command creates the branch and switches the repository to it.

3. Verify you are in your new branch.

        $ git branch
        * dry-run-test
          master

    The current branch has an * (asterisk) marker. So, these results shows you
    are on the right branch. 

4. Create a `TEST.md` file in the repository's root.

        $ touch TEST.md
	
5. Edit the file and add your email and location.

    ![Add your information](/project/images/contributor-edit.png)

    You can use any text editor you are comfortable with.

6. Save and close the file.

7. Check the status of your branch. 

        $ git status
        On branch dry-run-test
        Untracked files:
          (use "git add <file>..." to include in what will be committed)
    
            TEST.md
    
        nothing added to commit but untracked files present (use "git add" to track)

	You've only changed the one file. It is untracked so far by git.

8. Add your file.

        $ git add TEST.md

    That is the only _staged_ file. Stage is fancy word for work that Git is
    tracking.

9. Sign and commit your change.

        $ git commit -s -m "Making a dry run test."
        [dry-run-test 6e728fb] Making a dry run test
         1 file changed, 1 insertion(+)
         create mode 100644 TEST.md

    Commit messages should have a short summary sentence of no more than 50
    characters. Optionally, you can also include a more detailed explanation
    after the summary. Separate the summary from any explanation with an empty
    line.

8. Push your changes to GitHub.

        $ git push --set-upstream origin dry-run-test
        Username for 'https://github.com': moxiegirl
        Password for 'https://moxiegirl@github.com': 

    Git prompts you for your GitHub username and password. Then, the command
    returns a result.

        Counting objects: 13, done.
        Compressing objects: 100% (2/2), done.
        Writing objects: 100% (3/3), 320 bytes | 0 bytes/s, done.
        Total 3 (delta 1), reused 0 (delta 0)
        To https://github.com/moxiegirl/docker.git
         * [new branch]      dry-run-test -> dry-run-test
        Branch dry-run-test set up to track remote branch dry-run-test from origin.

9. Open your browser to GitHub.

10. Navigate to your Docker fork.

11. Make sure the `dry-run-test` branch exists, that it has your commit, and the
commit is signed.

    ![Branch Signature](/project/images/branch-sig.png)

## Where to go next

Congratulations, you have finished configuring both your local host environment
and Git for contributing. In the next section you'll [learn how to set up and
work in a Docker development container](/project/set-up-dev-env/).
