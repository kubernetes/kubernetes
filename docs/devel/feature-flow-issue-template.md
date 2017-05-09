Thanks for wanting to add a feature to Kubernetes!  You will be responsible for guiding
your feature through completion, and asking the right people for approvals.  

Large features typically go through three stages: [Alpha, Beta, and Stable](https://github.com/kubernetes/kubernetes/blob/master/docs/api.md#api-versioning)
Each stage requires various approvals from various teams.

You can use the checkboxes below to
track your progress.  You can delete the text above once you have read it.  Please keep the checklist.

- [ ] Before Alpha
    - [ ] Design Approval
      - [ ] < 1 page PR description of the design. [advice for design description]  **PR-NUMBER**
      - [ ] Design doc under [docs/proposals](https://github.com/kubernetes/kubernetes/docs/proposals).  **PR-NUMBER**
      - [ ] Initial API review (if API).  Maybe same PR as design doc. **PR-NUMBER**
        -  Any code that changes an API (`/pkg/apis/...`)
        -  **cc @kubernetes/api**
    - [ ] Write (code + tests + docs) then get them merged.  **ALL-PR-NUMBERS**
      - [ ] **Code needs to be disabled by default.**   Verified by code OWNERS
      - [ ] Minimal testing
      - [ ] Minimal docs
        - cc @kubernetes/docs on docs PR
        - **cc @kubernetes/feature-reviewers on this issue** to get approval before checking this off
        - New apis: *Glossary Section Item* in the docs repo: kubernetes/kubernetes.github.io
      - [ ] Update release notes
- [ ] Before Beta
  - [ ] Testing is sufficient for beta
  - [ ] User docs with tutorials
        - *Updated walkthrough / tutorial* in the docs repo: kubernetes/kubernetes.github.io
        - cc @kubernetes/docs on docs PR
        - **cc @kubernetes/feature-reviewers on this issue** to get approval before checking this off
  - [ ] Thorough API review
    - **cc @kubernetes/api**
- [ ] Before Stable
  - [ ] docs/proposals/foo.md moved to docs/design/foo.md 
        - **cc @kubernetes/feature-reviewers on this issue** to get approval before checking this off
  - [ ] Soak, load testing 			
  - [ ] detailed user docs and examples
    - **cc @kubernetes/docs**
    - **cc @kubernetes/feature-reviewers on this issue** to get approval before checking this off

*FEATURE_STATUS is used for feature tracking and to be updated by @kubernetes/feature-reviewers.*
**FEATURE_STATUS: IN_DEVELOPMENT**

More advice:

Design
   - Once you get LGTM from a *@kubernetes/feature-reviewers* member, you can check this checkbox, and the reviewer will apply the "design-complete" label.
 
Coding
  - Use as many PRs as you need.  Write tests in the same or different PRs, as is convenient for you.
  - As each PR is merged, add a comment to this issue referencing the PRs.  Code goes in the http://github.com/kubernetes/kubernetes repository,
        and sometimes http://github.com/kubernetes/contrib, or other repos.
  - When you are done with the code, apply the "code-complete" label.
  - When the feature has user docs, please add a comment mentioning @kubernetes/feature-reviewers and they will
        check that the code matches the proposed feature and design, and that everything is done, and that there is adequate
        testing.  They won't do detailed code review: that already happened when your PRs were reviewed.
        When that is done, you can check this box and the reviewer will apply the "code-complete" label.

Docs
  - [ ] Write user docs and get them merged in.
  - User docs go into http://github.com/kubernetes/kubernetes.github.io.
  - When the feature has user docs, please add a comment mentioning @kubernetes/docs.
  - When you get LGTM, you can check this checkbox, and the reviewer will apply the "docs-complete" label.
