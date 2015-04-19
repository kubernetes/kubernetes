# Chokidar 0.8.2 (26 March 2014)
* Fixed npm issues related to fsevents dep.
* Updated fsevents to 0.2.

# Chokidar 0.8.1 (16 December 2013)
* Optional deps are now truly optional on windows and
  linux.
* Rewritten in JS, again.
* Fixed some FSEvents-related bugs.

# Chokidar 0.8.0 (29 November 2013)
* Added ultra-fast low-CPU OS X file watching with FSEvents.
  It is enabled by default.
* Added `addDir` and `unlinkDir` events.
* Polling is now disabled by default on all platforms.

# Chokidar 0.7.1 (18 November 2013)
* `Watcher#close` now also removes all event listeners.

# Chokidar 0.7.0 (22 October 2013)
* When `options.ignored` is two-argument function, it will
  also be called after stating the FS, with `stats` argument.
* `unlink` is no longer emitted on directories.

# Chokidar 0.6.3 (12 August 2013)
* Added `usePolling` option (default: `true`).
  When `false`, chokidar will use `fs.watch` as backend.
  `fs.watch` is much faster, but not like super reliable.

# Chokidar 0.6.2 (19 March 2013)
* Fixed watching initially empty directories with `ignoreInitial` option.

# Chokidar 0.6.1 (19 March 2013)
* Added node.js 0.10 support.

# Chokidar 0.6.0 (10 March 2013)
* File attributes (stat()) are now passed to `add` and `change` events
  as second arguments.
* Changed default polling interval for binary files to 300ms.

# Chokidar 0.5.3 (13 January 2013)
* Removed emitting of `change` events before `unlink`.

# Chokidar 0.5.2 (13 January 2013)
* Removed postinstall script to prevent various npm bugs.

# Chokidar 0.5.1 (6 January 2013)
* When starting to watch non-existing paths, chokidar will no longer throw
ENOENT error.
* Fixed bug with absolute path.

# Chokidar 0.5.0 (9 December 2012)
* Added a bunch of new options:
    * `ignoreInitial` that allows to ignore initial `add` events.
    * `ignorePermissionErrors` that allows to ignore ENOENT etc perm errors.
    * `interval` and `binaryInterval` that allow to change default
    fs polling intervals.

# Chokidar 0.4.0 (26 July 2012)
* Added `all` event that receives two args (event name and path) that
combines `add`, `change` and `unlink` events.
* Switched to `fs.watchFile` on node.js 0.8 on windows.
* Files are now correctly unwatched after unlink.

# Chokidar 0.3.0 (24 June 2012)
* `unlink` event are no longer emitted for directories, for consistency
with `add`.

# Chokidar 0.2.6 (8 June 2012)
* Prevented creating of duplicate 'add' events.

# Chokidar 0.2.5 (8 June 2012)
* Fixed a bug when new files in new directories hadn't been added.

# Chokidar 0.2.4 (7 June 2012)
* Fixed a bug when unlinked files emitted events after unlink.

# Chokidar 0.2.3 (12 May 2012)
* Fixed watching of files on windows.

# Chokidar 0.2.2 (4 May 2012)
* Fixed watcher signature.

# Chokidar 0.2.1 (4 May 2012)
* Fixed invalid API bug when using `watch()`.

# Chokidar 0.2.0 (4 May 2012)
* Rewritten in js.

# Chokidar 0.1.1 (26 April 2012)
* Changed api to `chokidar.watch()`.
* Fixed compilation on windows.

# Chokidar 0.1.0 (20 April 2012)
* Initial release.
