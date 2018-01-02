---
title: Dockercon 2017 Summit
layout: home
---

# Dockercon 2017 Summit

This year at Dockercon US 2017 we will be having a containerd Summit on Thursday morning the week of the conference.  

We are going to change the format slightly compared to the previous summit that was held in February.  We will be allocating more time to the breakout sessions and less time to static talks.  However, the group will be much larger than the previous summit so this document serves as a way to add discussion points for the breakout sessions. 

If you would like to add a discussion point to the agenda, submit a PR adding it to the list below.  A simple one line sentence is enough or expand if needed. 

If you have not signed up to attend the summit you can do so in this [form](https://docs.google.com/forms/d/e/1FAIpQLScNkLm984ABbFChPh02uJR2lJ6y1AXjFaDITCaxTFL-sHhPwQ/viewform).

## Discussion Points

The following are proposed discussion points for the containerd summit at Dockercon US 2017:


* Since containerd is one of the bottom bricks in the stack, how can we setup automated integration tests for consumers of containerd? 
* We'd like to propose an Authorization plugin to containerd that would allow an external component to police events like container start & stop (and have a discussion about the best way to go about it)
* Should containerd provide image filesystem metrics? If yes, what metrics should be included? How to implement that?
* Support for disk quotas: How? What is the role of containerd? How is it going to be integrated with volume managers that want to be in the same quota group?
* Checkpoint/Restore: how can we support more use cases? One of the big issues here is the large number of options that can be passed to CRIU.
* How to support multi-OS docker images, for example, Linux Vs Windows using one graph driver plugin properly? 
