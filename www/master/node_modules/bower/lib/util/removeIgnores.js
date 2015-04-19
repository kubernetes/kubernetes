var path = require('path');
var rimraf = require('rimraf');
var fstreamIgnore = require('fstream-ignore');
var mout = require('mout');
var Q = require('q');

function removeIgnores(dir, meta) {
    var reader;
    var applyIgnores;
    var deferred = Q.defer();
    var ignored = [];
    var nonIgnored = ['bower.json'];

    // Don't ignore main files
    nonIgnored = nonIgnored.concat(meta.main || []);

    nonIgnored = nonIgnored.map(function (file) {
        return path.join(dir, file);
    });

    reader = fstreamIgnore({
        path: dir,
        type: 'Directory'
    });

    reader.addIgnoreRules(meta.ignore || []);

    // Monkey patch applyIgnores such that we get hold of all ignored files
    applyIgnores = reader.applyIgnores;
    reader.applyIgnores = function (entry) {
        var ret = applyIgnores.apply(this, arguments);

        if (!ret) {
            ignored.push(path.join(dir, entry));
        }

        return ret;
    };

    reader
    .on('child', function (entry) {
        nonIgnored.push(entry.path);
    })
    .on('error', deferred.reject)
    .on('end', function () {
        var promises;

        // Ensure that we are not ignoring files that should not be ignored!
        ignored = mout.array.unique(ignored);
        ignored = ignored.filter(function (file) {
            return nonIgnored.indexOf(file) === -1;
        });

        // Delete all the ignored files
        promises = ignored.map(function (file) {
            return Q.nfcall(rimraf, file);
        });

        return Q.all(promises)
        .then(deferred.resolve, deferred.reject);
    });

    return deferred.promise;
}

module.exports = removeIgnores;
