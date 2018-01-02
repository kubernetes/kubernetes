Triaging of issues
------------------

Triage provides an important way to contribute to an open source project.  Triage helps ensure issues resolve quickly by:

- Describing the issue's intent and purpose is conveyed precisely. This is necessary because it can be difficult for an issue to explain how an end user experiences a problem and what actions they took.
- Giving a contributor the information they need before they commit to resolving an issue.
- Lowering the issue count by preventing duplicate issues.
- Streamlining the development process by preventing duplicate discussions.

If you don't have time to code, consider helping with triage. The community will thank you for saving them time by spending some of yours.

### 1. Ensure the issue contains basic information

Before triaging an issue very far, make sure that the issue's author provided the standard issue information. This will help you make an educated recommendation on how this to categorize the issue. Standard information that *must* be included in most issues are things such as:

-   the output of `docker version`
-   the output of `docker info`
-   the output of `uname -a`
-   a reproducible case if this is a bug, Dockerfiles FTW
-   host distribution and version ( ubuntu 14.04, RHEL, fedora 23 )
-   page URL if this is a docs issue or the name of a man page

Depending on the issue, you might not feel all this information is needed. Use your best judgement.  If you cannot triage an issue using what its author provided, explain kindly to the author that they must provide the above information to clarify the problem.

If the author provides the standard information but you are still unable to triage the issue, request additional information. Do this kindly and politely because you are asking for more of the author's time.

If the author does not respond requested information within the timespan of a week, close the issue with a kind note stating that the author can request for the issue to be
reopened when the necessary information is provided.

### 2. Classify the Issue

An issue can have multiple of the following labels. Typically, a properly classified issue should
have:

- One label identifying its kind (`kind/*`).
- One or multiple labels identifying the functional areas of interest (`area/*`).
- Where applicable, one label categorizing its difficulty (`exp/*`).

#### Issue kind

| Kind             | Description                                                                                                                     |
|------------------|---------------------------------------------------------------------------------------------------------------------------------|
| kind/bug         | Bugs are bugs. The cause may or may not be known at triage time so debugging should be taken account into the time estimate.    |
| kind/enhancement | Enhancements are not bugs or new features but can drastically improve usability or performance of a project component.           |
| kind/feature     | Functionality or other elements that the project does not currently support.  Features are new and shiny.                       |
| kind/question    | Contains a user or contributor question requiring a response.                                                                   |

#### Functional area

| Area                      |
|---------------------------|
| area/api                  |
| area/builder              |
| area/bundles              |
| area/cli                  |
| area/daemon               |
| area/distribution         |
| area/docs                 |
| area/kernel               |
| area/logging              |
| area/networking           |
| area/plugins              |
| area/project              |
| area/runtime              |
| area/security             |
| area/security/apparmor    |
| area/security/seccomp     |
| area/security/selinux     |
| area/security/trust       |
| area/storage              |
| area/storage/aufs         |
| area/storage/btrfs        |
| area/storage/devicemapper |
| area/storage/overlay      |
| area/storage/zfs          |
| area/swarm                |
| area/testing              |
| area/volumes              |

#### Platform

| Platform                  |
|---------------------------|
| platform/arm              |
| platform/darwin           |
| platform/ibm-power        |
| platform/ibm-z            |
| platform/windows          |

#### Experience level

Experience level is a way for a contributor to find an issue based on their
skill set.  Experience types are applied to the issue or pull request using
labels.

| Level            | Experience level guideline                                                                                                                                                  |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| exp/beginner     | New to Docker, and possibly Golang, and is looking to help while learning the basics.                                                                                       |
| exp/intermediate | Comfortable with golang and understands the core concepts of Docker and looking to dive deeper into the project.                                                            |
| exp/expert       | Proficient with Docker and Golang and has been following, and active in, the community to understand the rationale behind design decisions and where the project is headed. |

As the table states, these labels are meant as guidelines. You might have
written a whole plugin for Docker in a personal project and never contributed to
Docker. With that kind of experience, you could take on an <strong
class="gh-label expert">exp/expert</strong> level task.

#### Triage status

To communicate the triage status with other collaborators, you can apply status
labels to issues. These labels prevent duplicating effort.

| Status                        | Description                                                                                                                                                                 |
|-------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| status/confirmed              | You triaged the issue, and were able to reproduce the issue. Always leave a comment how you reproduced, so that the person working on resolving the issue has a way to set up a test-case.
| status/accepted               | Apply to enhancements / feature requests that we think are good to have. Adding this label helps contributors find things to work on.
| status/more-info-needed       | Apply this to issues that are missing information (e.g. no `docker version` or `docker info` output, or no steps to reproduce), or require feedback from the reporter. If the issue is not updated after a week, it can generally be closed.
| status/needs-attention        | Apply this label if an issue (or PR) needs more eyes.

### 3. Prioritizing issue

When, and only when, an issue is attached to a specific milestone, the issue can be labeled with the
following labels to indicate their degree of priority (from more urgent to less urgent).

| Priority    | Description                                                                                                                       |
|-------------|-----------------------------------------------------------------------------------------------------------------------------------|
| priority/P0 | Urgent: Security, critical bugs, blocking issues. P0 basically means drop everything you are doing until this issue is addressed. |
| priority/P1 | Important: P1 issues are a top priority and a must-have for the next release.                                                     |
| priority/P2 | Normal priority: default priority applied.                                                                                        |
| priority/P3 | Best effort: those are nice to have / minor issues.                                                                               |

And that's it. That should be all the information required for a new or existing contributor to come in a resolve an issue.
