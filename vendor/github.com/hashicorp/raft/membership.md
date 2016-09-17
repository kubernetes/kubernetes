Simon (@superfell) and I (@ongardie) talked through reworking this library's cluster membership changes last Friday. We don't see a way to split this into independent patches, so we're taking the next best approach: submitting the plan here for review, then working on an enormous PR. Your feedback would be appreciated. (@superfell is out this week, however, so don't expect him to respond quickly.)

These are the main goals:
 - Bringing things in line with the description in my PhD dissertation;
 - Catching up new servers prior to granting them a vote, as well as allowing permanent non-voting members; and
 - Eliminating the `peers.json` file, to avoid issues of consistency between that and the log/snapshot.

## Data-centric view

We propose to re-define a *configuration* as a set of servers, where each server includes an address (as it does today) and a mode that is either:
 - *Voter*: a server whose vote is counted in elections and whose match index is used in advancing the leader's commit index.
 - *Nonvoter*: a server that receives log entries but is not considered for elections or commitment purposes.
 - *Staging*: a server that acts like a nonvoter with one exception: once a staging server receives enough log entries to catch up sufficiently to the leader's log, the leader will invoke a  membership change to change the staging server to a voter.

All changes to the configuration will be done by writing a new configuration to the log. The new configuration will be in affect as soon as it is appended to the log (not when it is committed like a normal state machine command). Note that, per my dissertation, there can be at most one uncommitted configuration at a time (the next configuration may not be created until the prior one has been committed). It's not strictly necessary to follow these same rules for the nonvoter/staging servers, but we think its best to treat all changes uniformly.

Each server will track two configurations:
 1. its *committed configuration*: the latest configuration in the log/snapshot that has been committed, along with its index.
 2. its *latest configuration*: the latest configuration in the log/snapshot (may be committed or uncommitted), along with its index.

When there's no membership change happening, these two will be the same. The latest configuration is almost always the one used, except:
 - When followers truncate the suffix of their logs, they may need to fall back to the committed configuration.
 - When snapshotting, the committed configuration is written, to correspond with the committed log prefix that is being snapshotted.


## Application API

We propose the following operations for clients to manipulate the cluster configuration:
 - AddVoter: server becomes staging unless voter,
 - AddNonvoter: server becomes nonvoter unless staging or voter,
 - DemoteVoter: server becomes nonvoter unless absent,
 - RemovePeer: server removed from configuration,
 - GetConfiguration: waits for latest config to commit, returns committed config.

This diagram, of which I'm quite proud, shows the possible transitions:
```
+-----------------------------------------------------------------------------+
|                                                                             |
|                      Start ->  +--------+                                   |
|            ,------<------------|        |                                   |
|           /                    | absent |                                   |
|          /       RemovePeer--> |        | <---RemovePeer                    |
|         /            |         +--------+               \                   |
|        /             |            |                      \                  |
|   AddNonvoter        |         AddVoter                   \                 |
|       |       ,->---' `--<-.      |                        \                |
|       v      /              \     v                         \               |
|  +----------+                +----------+                    +----------+   |
|  |          | ---AddVoter--> |          | -log caught up --> |          |   |
|  | nonvoter |                | staging  |                    |  voter   |   |
|  |          | <-DemoteVoter- |          |                 ,- |          |   |
|  +----------+         \      +----------+                /   +----------+   |
|                        \                                /                   |
|                         `--------------<---------------'                    |
|                                                                             |
+-----------------------------------------------------------------------------+
```

While these operations aren't quite symmetric, we think they're a good set to capture
the possible intent of the user. For example, if I want to make sure a server doesn't have a vote, but the server isn't part of the configuration at all, it probably shouldn't be added as a nonvoting server.

Each of these application-level operations will be interpreted by the leader and, if it has an effect, will cause the leader to write a new configuration entry to its log. Which particular application-level operation caused the log entry to be written need not be part of the log entry.

## Code implications

This is a non-exhaustive list, but we came up with a few things:
- Remove the PeerStore: the `peers.json` file introduces the possibility of getting out of sync with the log and snapshot, and it's hard to maintain this atomically as the log changes. It's not clear whether it's meant to track the committed or latest configuration, either.
- Servers will have to search their snapshot and log to find the committed configuration and the latest configuration on startup.
- Bootstrap will no longer use `peers.json` but should initialize the log or snapshot with an application-provided configuration entry.
- Snapshots should store the index of their configuration along with the configuration itself. In my experience with LogCabin, the original log index of the configuration is very useful to include in debug log messages.
- As noted in hashicorp/raft#84, configuration change requests should come in via a separate channel, and one may not proceed until the last has been committed.
- As to deciding when a log is sufficiently caught up, implementing a sophisticated algorithm *is* something that can be done in a separate PR. An easy and decent placeholder is: once the staging server has reached 95% of the leader's commit index, promote it.

## Feedback

Again, we're looking for feedback here before we start working on this. Here are some questions to think about:
 - Does this seem like where we want things to go?
 - Is there anything here that should be left out?
 - Is there anything else we're forgetting about?
 - Is there a good way to break this up?
 - What do we need to worry about in terms of backwards compatibility?
 - What implication will this have on current tests?
 - What's the best way to test this code, in particular the small changes that will be sprinkled all over the library?
