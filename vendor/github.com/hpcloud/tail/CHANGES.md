# API v1 (gopkg.in/hpcloud/tail.v1)

## April, 2016

* Migrated to godep, as depman is not longer supported
* Introduced golang vendoring feature
* Fixed issue [#57](https://github.com/hpcloud/tail/issues/57) related to reopen deleted file 

## July, 2015

* Fix inotify watcher leak; remove `Cleanup` (#51)

# API v0 (gopkg.in/hpcloud/tail.v0)

## June, 2015

* Don't return partial lines (PR #40)
* Use stable version of fsnotify (#46)

## July, 2014

* Fix tail for Windows (PR #36)

## May, 2014

* Improved rate limiting using leaky bucket (PR #29)
* Fix odd line splitting (PR #30)

## Apr, 2014

* LimitRate now discards read buffer (PR #28)
* allow reading of longer lines if MaxLineSize is unset (PR #24)
* updated deps.json to latest fsnotify (441bbc86b1)

## Feb, 2014

* added `Config.Logger` to suppress library logging

## Nov, 2013

* add Cleanup to remove leaky inotify watches (PR #20)

## Aug, 2013

* redesigned Location field (PR #12)
* add tail.Tell (PR #14)

## July, 2013

* Rate limiting (PR #10)

## May, 2013

* Detect file deletions/renames in polling file watcher (PR #1)
* Detect file truncation
* Fix potential race condition when reopening the file (issue 5)
* Fix potential blocking of `tail.Stop` (issue 4)
* Fix uncleaned up ChangeEvents goroutines after calling tail.Stop
* Support Follow=false

## Feb, 2013

* Initial open source release
