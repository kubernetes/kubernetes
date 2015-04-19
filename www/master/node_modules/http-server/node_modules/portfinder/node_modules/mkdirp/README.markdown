mkdirp
======

Like `mkdir -p`, but in node.js!

Example
=======

pow.js
------
    var mkdirp = require('mkdirp');
    
    mkdirp('/tmp/foo/bar/baz', 0755, function (err) {
        if (err) console.error(err)
        else console.log('pow!')
    });

Output
    pow!

And now /tmp/foo/bar/baz exists, huzzah!
