# Project Governance

This document defines the governance process for the CEL language. CEL is
Google-developed, but openly governed. Major contributors to the CEL
specification and its corresponding implementations constitute the CEL
Language Council. New members may be added by a unanimous vote of the
Council.

The MAINTAINERS.md file lists the members of the CEL Language Council, and
unofficially indicates the "areas of expertise" of each member with respect
to the publicly available CEL repos.

## Code Changes

Code changes must follow the standard pull request (PR) model documented in the
CONTRIBUTING.md for each CEL repo. All fixes and features must be reviewed by a
maintainer. The maintainer reserves the right to request that any feature
request (FR) or PR be reviewed by the language council.

## Syntax and Semantic Changes

Syntactic and semantic changes must be reviewed by the CEL Language Council.
Maintainers may also request language council review at their discretion.

The review process is as follows:

- Create a Feature Request in the CEL-Spec repo. The feature description will
  serve as an abstract for the detailed design document.
- Co-develop a design document with the Language Council.
- Once the proposer gives the design document approval, the document will be
  linked to the FR in the CEL-Spec repo and opened for comments to members of
  the cel-lang-discuss@googlegroups.com.
- The Language Council will review the design doc at the next council meeting
  (once every three weeks) and the council decision included in the document.

If the proposal is approved, the spec will be updated by a maintainer (if
applicable) and a rationale will be included in the CEL-Spec wiki to ensure
future developers may follow CEL's growth and direction over time.

Approved proposals may be implemented by the proposer or by the maintainers as
the parties see fit. At the discretion of the maintainer, changes from the
approved design are permitted during implementation if they improve the user
experience and clarity of the feature.
