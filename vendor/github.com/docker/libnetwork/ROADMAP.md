# Roadmap

This document defines the high-level goals of the libnetwork project. See [Project Planning](#project-planning) for information on Releases.

## Long-term Goal

libnetwork project will follow Docker and Linux philosophy of delivering small, highly modular and composable tools that work well independently. 
libnetwork aims to satisfy that composable need for Networking in Containers.

## Short-term Goals

- Modularize the networking logic in Docker Engine and libcontainer in to a single, reusable library
- Replace the networking subsystem of Docker Engine, with libnetwork
- Define a flexible model that allows local and remote drivers to provide networking to containers
- Provide a stand-alone tool "dnet" for managing and testing libnetwork

Project Planning
================

[Project Pages](https://github.com/docker/libnetwork/wiki) define the goals for each Milestone and identify the release-relationship to the Docker Platform.
