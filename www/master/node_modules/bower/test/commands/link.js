var expect = require('expect.js');
var helpers = require('../helpers');

var link = helpers.command('link');

describe('bower link', function () {

    var package = new helpers.TempDir({
        'bower.json': {
            name: 'package',
        },
        'index.js': 'Hello World!'
    });

    var otherPackage = new helpers.TempDir({
        'bower.json': {
            name: 'package2',
        },
        'index.js': 'Welcome World!'
    });

    var linksDir = new helpers.TempDir();

    beforeEach(function() {
        package.prepare();
        otherPackage.prepare();
        linksDir.prepare();
    });

    it('correctly reads arguments', function() {
        expect(link.readOptions(['jquery', 'angular']))
        .to.eql(['jquery', 'angular']);
    });

    it('creates self link', function () {
        return helpers.run(link, [undefined, undefined,
            {
                cwd: package.path,
                storage: {
                    links: linksDir.path
                }
            }
        ]).then(function() {
            expect(linksDir.read('package/index.js'))
            .to.be('Hello World!');
        });
    });

    it('creates inter-link', function () {
        return helpers.run(link, [undefined, undefined,
            {
                cwd: package.path,
                storage: {
                    links: linksDir.path
                }
            }
        ]).then(function () {
            return helpers.run(link, ['package', undefined,
                {
                    cwd: otherPackage.path,
                    storage: {
                        links: linksDir.path
                    }
                }
            ]);
        }).then(function() {
            expect(otherPackage.read('bower_components/package/index.js'))
            .to.be('Hello World!');
        });
    });

    it('creates inter-link with custom local name', function () {
        return helpers.run(link, [undefined, undefined,
            {
                cwd: package.path,
                storage: {
                    links: linksDir.path
                }
            }
        ]).then(function () {
            return helpers.run(link, ['package', 'local',
                {
                    cwd: otherPackage.path,
                    storage: {
                        links: linksDir.path
                    }
                }
            ]);
        }).then(function() {
            expect(otherPackage.read('bower_components/local/index.js'))
            .to.be('Hello World!');
        });
    });

    it('errors on unexising package', function () {
        return helpers.run(link, ['package', 'local',
            {
                cwd: otherPackage.path,
                storage: {
                    links: linksDir.path
                }
            }
        ]).then(function() {
            throw 'Should fail creating a link!';
        }).fail(function(reason) {
            expect(reason.code).to.be('ENOENT');
            expect(reason.message).to.be('Failed to create link to package');
        });
    });
});
