# mkpath

Make all directories in a path, like `mkdir -p`.

## How to use

    var mkpath = require('mkpath');
    
    mkpath('red/green/violet', function (err) {
        if (err) throw err;
        console.log('Directory structure red/green/violet created');
    });
    
    mkpath.sync('/tmp/blue/orange', 0700);

### mkpath(path, [mode = 0777 & (~process.umask()),] [callback])

Create all directories that don't exist in `path` with permissions `mode`. When finished, `callback(err)` fires with the error, if any.

### mkpath.sync(path, [mode = 0777 & (~process.umask())]);

Synchronous version of the same. Throws error, if any.

## License

This software is released under the [MIT license](http://www.opensource.org/licenses/MIT).

