Triaging of issues
------------------

Triage provides an important way to contribute to an open source project.  Triage helps ensure issues resolve quickly by:  

- Describing the issue's intent and purpose is conveyed precisely. This is necessary because it can be difficult for an issue to explain how an end user experiences an problem and what actions they took. 

- Giving a contributor the information they need before they commit to resolving an issue. 

- Lowering the issue count by preventing duplicate issues.

- Streamlining the development process by preventing duplicate discussions.

If you don't have time to code, consider helping with triage. The community will thank you for saving them time by spending some of yours.

### Step 1: Ensure the issue contains basic information

Before triaging an issue very far, make sure that the issue's author provided the standard issue information. This will help you make an educated recommendation on how this to categorize the issue. Standard information that *must* be included in most issues are things such as:

-   the output of `docker version`
-   the output of `docker info`
-   the output of `uname -a`
-   a reproducible case if this is a bug, Dockerfiles FTW
-   host distribution and version ( ubuntu 14.04, RHEL, fedora 21 )
-   page URL if this is a docs issue or the name of a man page 

Depending on the issue, you might not feel all this information is needed. Use your best judgement.  If you cannot triage an issue using what its author provided, explain kindly to the author that they must provide the above information to clarify the problem. 

If the author provides the standard information but you are still unable to triage the issue, request additional information. Do this kindly and politely because you are asking for more of the author's time.

If the author does not respond requested information within the timespan of a week, close the issue with a kind note stating that the author can request for the issue to be
reopened when the necessary information is provided.

### Step 2: Apply the template

When triaging, use the standard template below. You should cut and place the template in the issue's description. 
The template helps other reviewers find key information in an issue. For example, using a template saves a 
potential contributor from wading though 100s of comments to find a proposed solution at the very end.  When adding 
the template to the issue's description also add any required labels to the issue for the classification and difficulty.

Here is a sample summary for an [issue](https://github.com/docker/docker/issues/10545).

```
**Summary**: docker rm can return a non-zero exit code if the container does not
exist and it is not easy to parse the error message.

**Proposed solution**:

docker rm should have consistent exit codes for different types of errors so
that the user can easily script and know the reason why the command failed. 

```

### Step 3: Classify the Issue

Classifications help both to inform readers about an issue's priority and how to resolve it.
This is also helpful for identifying new, critical issues.  "Kinds of" are
applied to the issue or pull request using labels.  You can apply one or more labels.


Kinds of classifications:

| Kind             | Description                                                                                                                     |
|------------------|---------------------------------------------------------------------------------------------------------------------------------|
| kind/enhancement | Enhancement are not bugs or new features but can drastically improve usability or performance of a project component.           |
| kind/cleanup     | Refactoring code or otherwise clarifying documentation.                                                                         |
| kind/content     | Content that is not documentation such as help or error messages.                                                               |
| kind/graphics    | Work involving graphics skill                                                                                                   |
| kind/regression  | Regressions are usually easy fixes as hopefully the action worked previously and git history can be used to propose a solution. |
| kind/bug         | Bugs are bugs. The cause may or may not be known at triage time so debugging should be taken account into the time estimate.    |
| kind/feature     | Functionality or other elements that the project does not currently support.  Features are new and shinny.                      |
| kind/question    | Contains a user or contributor question requiring a response.                                                                   |
| kind/usecase     | A description of a user or contributor situation requiring a response perhaps in code or documentation.                         |
| kind/writing     | Writing documentation, man pages, articles, blogs, or other significant word-driven task.                                       |
| kind/test        | Tests or test infrastructure needs adding or updating.                                                                                                 |


Contributors can add labels by using a `+kind/bug` in an issue or pull request comment.  

### Step 4: Estimate the experience level required

Experience level is a way for a contributor to find an issue based on their
skill set.  Experience types are applied to the issue or pull request using
labels.

| Level            | Experience level guideline                                                                                               |
|------------------|--------------------------------------------------------------------------------------------------------------------------|
| exp/beginner     | You have made less than 10 contributions in your life time to any open source project.                                   |
| exp/novice       | You have made more than 10 contributions to an open source project or at least 5 contributions to Docker.                | 
| exp/proficient   | You have made more than 5 contributions to Docker which amount to at least 200 code lines or 1000 documentation lines.   | 
| exp/expert       | You have made less than 20 commits to Docker which amount to 500-1000 code lines or 1000-3000 documentation lines.       | 
| exp/master       | You have made more than 20 commits to Docker and greater than 1000 code lines or 3000 documentation lines.               | 

As the table states, these labels are meant as guidelines. You might have
written a whole plugin for Docker in a personal project and never contributed to
Docker. With that kind of experience, you could take on an <strong
class="gh-label expert">exp/expert</strong> or <strong class="gh-label
master">exp/master</strong> level task.

Contributors can add labels by using a `+exp/expert` format in issue comment.


And that's it. That should be all the information required for a new or existing contributor to come in an resolve an issue.

