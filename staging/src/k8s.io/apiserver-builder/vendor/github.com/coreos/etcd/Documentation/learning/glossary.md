# Glossary

This document defines the various terms used in etcd documentation, command line and source code.

## Node

Node is an instance of raft state machine.

It has a unique identification, and records other nodes' progress internally when it is the leader.

## Member

Member is an instance of etcd. It hosts a node, and provides service to clients.

## Cluster

Cluster consists of several members.

The node in each member follows raft consensus protocol to replicate logs. Cluster receives proposals from members, commits them and apply to local store.

## Peer

Peer is another member of the same cluster.

## Proposal

A proposal is a request (for example a write request, a configuration change request) that needs to go through raft protocol.

## Client

Client is a caller of the cluster's HTTP API.

## Machine (deprecated)

The alternative of Member in etcd before 2.0
