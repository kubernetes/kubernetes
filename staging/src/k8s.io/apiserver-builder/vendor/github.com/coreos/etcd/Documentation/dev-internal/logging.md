# Logging conventions

etcd uses the [capnslog][capnslog] library for logging application output categorized into *levels*. A log message's level is determined according to these conventions:

* Error: Data has been lost, a request has failed for a bad reason, or a required resource has been lost
  * Examples: 
    * A failure to allocate disk space for WAL

* Warning: (Hopefully) Temporary conditions that may cause errors, but may work fine. A replica disappearing (that may reconnect) is a warning.
  * Examples:
    * Failure to send raft message to a remote peer
    * Failure to receive heartbeat message within the configured election timeout

* Notice: Normal, but important (uncommon) log information.
  * Examples:
    * Add a new node into the cluster
    * Add a new user into auth subsystem

* Info: Normal, working log information, everything is fine, but helpful notices for auditing or common operations.
  * Examples:
    * Startup configuration
    * Start to do snapshot

* Debug: Everything is still fine, but even common operations may be logged, and less helpful but more quantity of notices.
  * Examples:
    * Send a normal message to a remote peer
    * Write a log entry to disk

[capnslog]: [https://github.com/coreos/pkg/tree/master/capnslog]
