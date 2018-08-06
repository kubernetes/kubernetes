## Version 1.3 (2016-12-01)

Changes:

 - Go 1.1 is no longer supported
 - Use decimals fields in MySQL to format time types (#249)
 - Buffer optimizations (#269)
 - TLS ServerName defaults to the host (#283)
 - Refactoring (#400, #410, #437)
 - Adjusted documentation for second generation CloudSQL (#485)
 - Documented DSN system var quoting rules (#502)
 - Made statement.Close() calls idempotent to avoid errors in Go 1.6+ (#512)

New Features:

 - Enable microsecond resolution on TIME, DATETIME and TIMESTAMP (#249)
 - Support for returning table alias on Columns() (#289, #359, #382)
 - Placeholder interpolation, can be actived with the DSN parameter `interpolateParams=true` (#309, #318, #490)
 - Support for uint64 parameters with high bit set (#332, #345)
 - Cleartext authentication plugin support (#327)
 - Exported ParseDSN function and the Config struct (#403, #419, #429)
 - Read / Write timeouts (#401)
 - Support for JSON field type (#414)
 - Support for multi-statements and multi-results (#411, #431)
 - DSN parameter to set the driver-side max_allowed_packet value manually (#489)
 - Native password authentication plugin support (#494, #524)

Bugfixes:

 - Fixed handling of queries without columns and rows (#255)
 - Fixed a panic when SetKeepAlive() failed (#298)
 - Handle ERR packets while reading rows (#321)
 - Fixed reading NULL length-encoded integers in MySQL 5.6+ (#349)
 - Fixed absolute paths support in LOAD LOCAL DATA INFILE (#356)
 - Actually zero out bytes in handshake response (#378)
 - Fixed race condition in registering LOAD DATA INFILE handler (#383)
 - Fixed tests with MySQL 5.7.9+ (#380)
 - QueryUnescape TLS config names (#397)
 - Fixed "broken pipe" error by writing to closed socket (#390)
 - Fixed LOAD LOCAL DATA INFILE buffering (#424)
 - Fixed parsing of floats into float64 when placeholders are used (#434)
 - Fixed DSN tests with Go 1.7+ (#459)
 - Handle ERR packets while waiting for EOF (#473)
 - Invalidate connection on error while discarding additional results (#513)
 - Allow terminating packets of length 0 (#516)


## Version 1.2 (2014-06-03)

Changes:

 - We switched back to a "rolling release". `go get` installs the current master branch again
 - Version v1 of the driver will not be maintained anymore. Go 1.0 is no longer supported by this driver
 - Exported errors to allow easy checking from application code
 - Enabled TCP Keepalives on TCP connections
 - Optimized INFILE handling (better buffer size calculation, lazy init, ...)
 - The DSN parser also checks for a missing separating slash
 - Faster binary date / datetime to string formatting
 - Also exported the MySQLWarning type
 - mysqlConn.Close returns the first error encountered instead of ignoring all errors
 - writePacket() automatically writes the packet size to the header
 - readPacket() uses an iterative approach instead of the recursive approach to merge splitted packets

New Features:

 - `RegisterDial` allows the usage of a custom dial function to establish the network connection
 - Setting the connection collation is possible with the `collation` DSN parameter. This parameter should be preferred over the `charset` parameter
 - Logging of critical errors is configurable with `SetLogger`
 - Google CloudSQL support

Bugfixes:

 - Allow more than 32 parameters in prepared statements
 - Various old_password fixes
 - Fixed TestConcurrent test to pass Go's race detection
 - Fixed appendLengthEncodedInteger for large numbers
 - Renamed readLengthEnodedString to readLengthEncodedString and skipLengthEnodedString to skipLengthEncodedString (fixed typo)


## Version 1.1 (2013-11-02)

Changes:

  - Go-MySQL-Driver now requires Go 1.1
  - Connections now use the collation `utf8_general_ci` by default. Adding `&charset=UTF8` to the DSN should not be necessary anymore
  - Made closing rows and connections error tolerant. This allows for example deferring rows.Close() without checking for errors
  - `[]byte(nil)` is now treated as a NULL value. Before, it was treated like an empty string / `[]byte("")`
  - DSN parameter values must now be url.QueryEscape'ed. This allows text values to contain special characters, such as '&'.
  - Use the IO buffer also for writing. This results in zero allocations (by the driver) for most queries
  - Optimized the buffer for reading
  - stmt.Query now caches column metadata
  - New Logo
  - Changed the copyright header to include all contributors
  - Improved the LOAD INFILE documentation
  - The driver struct is now exported to make the driver directly accessible
  - Refactored the driver tests
  - Added more benchmarks and moved all to a separate file
  - Other small refactoring

New Features:

  - Added *old_passwords* support: Required in some cases, but must be enabled by adding `allowOldPasswords=true` to the DSN since it is insecure
  - Added a `clientFoundRows` parameter: Return the number of matching rows instead of the number of rows changed on UPDATEs
  - Added TLS/SSL support: Use a TLS/SSL encrypted connection to the server. Custom TLS configs can be registered and used

Bugfixes:

  - Fixed MySQL 4.1 support: MySQL 4.1 sends packets with lengths which differ from the specification
  - Convert to DB timezone when inserting `time.Time`
  - Splitted packets (more than 16MB) are now merged correctly
  - Fixed false positive `io.EOF` errors when the data was fully read
  - Avoid panics on reuse of closed connections
  - Fixed empty string producing false nil values
  - Fixed sign byte for positive TIME fields


## Version 1.0 (2013-05-14)

Initial Release
