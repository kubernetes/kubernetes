var expect = require('expect.js');
var helpers = require('../../helpers');

var cacheList = helpers.command('cache/list');

describe('bower cache list', function () {

    var cacheDir = new helpers.TempDir({
        '87323d6d4e48be291a9616a033d4cc6c/1.3.8/.bower.json': {
            name: 'angular',
            version: '1.3.8'
        },
        '87323d6d4e48be291a9616a033d4cc6c/1.3.9/.bower.json': {
            name: 'angular',
            version: '1.3.9'
        },
        '9eaed103d6a7e78d91f673cfad796850/1.0.0/.bower.json': {
            name: 'jquery',
            version: '1.0.0'
        }
    });

    it('correctly reads arguments', function() {
        expect(cacheList.readOptions(['jquery', 'angular']))
        .to.eql([['jquery', 'angular'], {}]);
    });

    it('lists packages from cache', function () {
        cacheDir.prepare();

        return helpers.run(cacheList, [undefined, {}, {
            storage: {
                packages: cacheDir.path
            }
        }]).spread(function(result) {
            expect(result[0].canonicalDir)
            .to.be(cacheDir.getPath('87323d6d4e48be291a9616a033d4cc6c/1.3.8'));
            expect(result[0].pkgMeta.version).to.be('1.3.8');
            expect(result[1].pkgMeta.version).to.be('1.3.9');
            expect(result[2].pkgMeta.version).to.be('1.0.0');
        });

    });

    it('lists selected package names', function () {
        cacheDir.prepare();

        return helpers.run(cacheList, [['angular'], {}, {
            storage: {
                packages: cacheDir.path
            }
        }]).spread(function(result) {
            expect(result[0].canonicalDir)
            .to.be(cacheDir.getPath('87323d6d4e48be291a9616a033d4cc6c/1.3.8'));
            expect(result[0].pkgMeta.version).to.be('1.3.8');
            expect(result[1].pkgMeta.version).to.be('1.3.9');
        });

    });
});
