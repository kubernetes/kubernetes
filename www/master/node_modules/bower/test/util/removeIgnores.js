var expect = require('expect.js');
var helpers = require('../helpers');
var glob = require('glob');
var Q = require('q');

var removeIgnores = require('../../lib/util/removeIgnores');

describe('removeIgnores', function () {

    var tempDir = new helpers.TempDir({
        'bower.json': {},
        'index.js': 'Not to ignore',
        'node_modules/underscore/index.js': 'Should be ignored'
    });

    var ignoreTest = function(dir, meta, leftovers) {
        tempDir.prepare();

        var deferred = Q.defer();

        removeIgnores(dir, meta).then(function() {
            glob('**/*.*', { cwd: dir }, function(cb, files) {
                expect(files).to.eql(leftovers);
                deferred.resolve();
            });
        });

        return deferred.promise;
    };

    it('removes all files in directory', function () {
        return ignoreTest(tempDir.path,
            { ignore: [ 'node_modules/**/*' ] },
            [ 'bower.json', 'index.js' ]
        );
    });

    it('removes whole directory', function () {
        return ignoreTest(tempDir.path,
            { ignore: [ 'node_modules/' ] },
            [ 'bower.json', 'index.js' ]
        );
    });

    it('removes whole directory (no ending slash)', function () {
        return ignoreTest(tempDir.path,
            { ignore: [ 'node_modules' ] },
            [ 'bower.json', 'index.js' ]
        );
    });

    it('removes all but one file', function() {
        return ignoreTest(tempDir.path,
            { ignore: [ '**/*', '!bower.json' ] },
            [ 'bower.json' ]
        );
    });

    it('refuses to ignore bower.json', function() {
        return ignoreTest(tempDir.path,
            { ignore: [ '**/*', '!index.js' ] },
            [ 'bower.json', 'index.js' ]
        );
    });

    it('removes all but one file deep down the tree', function() {
        return ignoreTest(tempDir.path,
            { ignore: [ '**/*', '!node_modules/underscore/index.js' ] },
            [
                'bower.json',
                'node_modules/underscore/index.js'
            ]
        );
    });
});
