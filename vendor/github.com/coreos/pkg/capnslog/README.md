# capnslog, the CoreOS logging package

There are far too many logging packages out there, with varying degrees of licenses, far too many features (colorization, all sorts of log frameworks) or are just a pain to use (lack of `Fatalln()`?).
capnslog provides a simple but consistent logging interface suitable for all kinds of projects.

### Design Principles

##### `package main` is the place where logging gets turned on and routed

A library should not touch log options, only generate log entries. Libraries are silent until main lets them speak.

##### All log options are runtime-configurable. 

Still the job of `main` to expose these configurations. `main` may delegate this to, say, a configuration webhook, but does so explicitly. 

##### There is one log object per package. It is registered under its repository and package name.

`main` activates logging for its repository and any dependency repositories it would also like to have output in its logstream. `main` also dictates at which level each subpackage logs.

##### There is *one* output stream, and it is an `io.Writer` composed with a formatter.

Splitting streams is probably not the job of your program, but rather, your log aggregation framework. If you must split output streams, again, `main` configures this and you can write a very simple two-output struct that satisfies io.Writer.

Fancy colorful formatting and JSON output are beyond the scope of a basic logging framework -- they're application/log-collector dependant. These are, at best, provided as options, but more likely, provided by your application.

##### Log objects are an interface

An object knows best how to print itself. Log objects can collect more interesting metadata if they wish, however, because text isn't going away anytime soon, they must all be marshalable to text. The simplest log object is a string, which returns itself. If you wish to do more fancy tricks for printing your log objects, see also JSON output -- introspect and write a formatter which can handle your advanced log interface. Making strings is the only thing guaranteed.

##### Log levels have specific meanings:

  * Critical: Unrecoverable. Must fail.
  * Error: Data has been lost, a request has failed for a bad reason, or a required resource has been lost
  * Warning: (Hopefully) Temporary conditions that may cause errors, but may work fine. A replica disappearing (that may reconnect) is a warning.
  * Notice: Normal, but important (uncommon) log information.
  * Info: Normal, working log information, everything is fine, but helpful notices for auditing or common operations.
  * Debug: Everything is still fine, but even common operations may be logged, and less helpful but more quantity of notices.
  * Trace: Anything goes, from logging every function call as part of a common operation, to tracing execution of a query.

