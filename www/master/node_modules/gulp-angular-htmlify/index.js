'use strict';
var gutil = require('gulp-util');
var through = require('through2');

module.exports = function (params) {
    params = params || {};
    var verbose = Boolean(params.verbose);
    var htmlify = require('angular-html5')({
        customPrefixes: params.customPrefixes
    });

    return through.obj(function (file, enc, cb) {
        if (file.isNull()) {
            this.push(file);
            return cb();
        }

        if (file.isStream()) {
            this.emit('error', new gutil.PluginError('gulp-angular-htmlify', 'Streaming not supported'));
            return cb();
        }

        var data = file.contents.toString('utf8');
        //if ng-directives exist
        if (htmlify.test(data)) {
            //replace contents and assign them back to stream contents
            file.contents = new Buffer(htmlify.replace(data));
            if (verbose) {
                //get filename or unknown
                var filename = gutil.colors.magenta(file.path || 'unknown');
                gutil.log(gutil.colors.blue('angular-htmlify'), 'found and replaced ng-directives in ' + filename);
            }
        }

        //push back file to stream
        this.push(file);
        return cb();
    });
};
