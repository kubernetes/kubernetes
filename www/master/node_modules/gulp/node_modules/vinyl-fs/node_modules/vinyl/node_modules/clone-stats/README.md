# clone-stats [![Flattr this!](https://api.flattr.com/button/flattr-badge-large.png)](https://flattr.com/submit/auto?user_id=hughskennedy&url=http://github.com/hughsk/clone-stats&title=clone-stats&description=hughsk/clone-stats%20on%20GitHub&language=en_GB&tags=flattr,github,javascript&category=software)[![experimental](http://hughsk.github.io/stability-badges/dist/experimental.svg)](http://github.com/hughsk/stability-badges) #

Safely clone node's
[`fs.Stats`](http://nodejs.org/api/fs.html#fs_class_fs_stats) instances without
losing their class methods, i.e. `stat.isDirectory()` and co.

## Usage ##

[![clone-stats](https://nodei.co/npm/clone-stats.png?mini=true)](https://nodei.co/npm/clone-stats)

### `copy = require('clone-stats')(stat)` ###

Returns a clone of the original `fs.Stats` instance (`stat`).

## License ##

MIT. See [LICENSE.md](http://github.com/hughsk/clone-stats/blob/master/LICENSE.md) for details.
