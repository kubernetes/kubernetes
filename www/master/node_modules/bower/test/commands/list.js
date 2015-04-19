var expect = require('expect.js');
var object = require('mout').object;
var path = require('path');

var helpers = require('../helpers');

var commands = {
    install: helpers.command('install'),
    list: helpers.command('list')
};

describe('bower list', function () {

    var tempDir = new helpers.TempDir();

    var gitPackage = new helpers.TempDir();

    var install = function(packages, options, config) {
        config = object.merge(config || {}, {
            cwd: tempDir.path
        });

        return helpers.run(commands.install, [packages, options, config]);
    };

    var list = function(options, config) {
        config = object.merge(config || {}, {
            cwd: tempDir.path
        });

        return helpers.run(commands.list, [options, config]);
    };

    it('correctly reads arguments', function() {
        expect(commands.list.readOptions(['-p', '-r']))
        .to.eql([{
            paths: true,
            relative: true
        }]);
    });

    it('correctly reads long arguments', function() {
        expect(commands.list.readOptions(['--paths', '--relative']))
        .to.eql([{
            paths: true,
            relative: true
        }]);
    });

    it('lists no packages when nothing installed', function () {
        tempDir.prepare();

        return list().spread(function(results) {
            expect(results).to.be.an(Object);
            expect(results.canonicalDir).to.equal(tempDir.path);
            expect(results.pkgMeta.dependencies).to.eql({});
            expect(results.pkgMeta.devDependencies).to.eql({});
            expect(results.dependencies).to.eql({});
            expect(results.nrDependants).to.eql(0);
            expect(results.versions).to.eql([]);
        });
    });

    it('lists 1 dependency when 1 local package installed', function () {

        var package = new helpers.TempDir({
            'bower.json': {
                name: 'package',
                main: 'test.txt'
            }
        }).prepare();
        package.prepare();

        return install([package.path]).then(function() {
            return list().spread(function(results) {
                expect(results).to.be.an(Object);
                expect(results.canonicalDir).to.equal(tempDir.path);
                expect(results.pkgMeta.dependencies).to.eql({
                    package: package.path + '#*'
                });
                expect(results.pkgMeta.devDependencies).to.eql({});
                expect(results.dependencies.package).to.be.an(Object);
                expect(results.dependencies.package.pkgMeta).to.be.an(Object);
                expect(results.dependencies.package.pkgMeta.main).to.equal('test.txt');
                expect(results.dependencies.package.canonicalDir).to.equal(
                    path.join(tempDir.path, 'bower_components/package')
                );
                expect(results.dependencies.package.dependencies).to.eql({});
                expect(results.dependencies.package.nrDependants).to.equal(1);
                expect(results.dependencies.package.versions).to.eql([]);
                expect(results.nrDependants).to.equal(0);
                expect(results.versions).to.eql([]);
            });
        });
    });

    it('lists 1 dependency with relative paths when 1 local package installed', function () {

        var package = new helpers.TempDir({
            'bower.json': {
                name: 'package',
                main: 'test.txt'
            }
        }).prepare();
        package.prepare();

        return install([package.path]).then(function() {
            return list({relative: true}).spread(function(results) {
                expect(results).to.be.an(Object);
                expect(results.canonicalDir).to.equal(tempDir.path);
                expect(results.dependencies).to.be.an(Object);
                expect(results.dependencies.package).to.be.an(Object);
                expect(results.dependencies.package.pkgMeta).to.be.an(Object);
                expect(results.dependencies.package.pkgMeta.main).to.equal('test.txt');
                expect(results.pkgMeta.dependencies).to.eql({
                    package: package.path + '#*'
                });
                expect(results.dependencies.package.canonicalDir).to.equal(
                    path.normalize('bower_components/package')
                );
            });
        });
    });

    it('lists 1 dependency with 1 source relative source mapping when 1 local package installed', function () {

        var package = new helpers.TempDir({
            'bower.json': {
                name: 'package',
                main: 'test.txt'
            }
        }).prepare();
        package.prepare();

        return install([package.path]).then(function() {
            return list({paths: true}).spread(function(results) {
                expect(results).to.be.an(Object);
                expect(results.package).to.equal(
                    'bower_components/package/test.txt'
                );
            });
        });
    });

    it('lists 1 dependency with 2 source relative source mapping when 1 local package installed', function () {

        var package = new helpers.TempDir({
            'bower.json': {
                name: 'package',
                main: ['test.txt', 'test2.txt']
            }
        }).prepare();
        package.prepare();

        return install([package.path]).then(function() {
            return list({paths: true}).spread(function(results) {
                expect(results).to.be.an(Object);
                expect(results.package).to.be.an(Object);
                expect(results.package).to.eql([
                    'bower_components/package/test.txt',
                    'bower_components/package/test2.txt'
                ]);
            });
        });
    });

    it('lists 1 dependency when 1 git package installed', function () {
        gitPackage.prepareGit({
            '1.0.0': {
                'bower.json': {
                    name: 'package',
                    main: 'test.txt'
                },
                'version.txt': '1.0.0'
            },
            '1.0.1': {
                'bower.json': {
                    name: 'package',
                    main: 'test2.txt'
                },
                'version.txt': '1.0.1'
            }
        });
        
        tempDir.prepare({
            'bower.json': {
                name: 'test',
                dependencies: {
                    package: gitPackage.path + '#1.0.0'
                }
            }
        });

        return install().then(function() {
            return list().spread(function(results) {
                expect(results).to.be.an(Object);
                expect(results.canonicalDir).to.equal(tempDir.path);
                expect(results.pkgMeta.dependencies).to.eql({
                    package: gitPackage.path + '#1.0.0'
                });
                expect(results.pkgMeta.devDependencies).to.eql({});
                expect(results.dependencies.package).to.be.an(Object);
                expect(results.dependencies.package.pkgMeta).to.be.an(Object);
                expect(results.dependencies.package.pkgMeta.main).to.equal('test.txt');
                expect(results.dependencies.package.canonicalDir).to.equal(
                    path.join(tempDir.path, 'bower_components/package')
                );
                expect(results.dependencies.package.dependencies).to.eql({});
                expect(results.dependencies.package.nrDependants).to.equal(1);
                expect(results.dependencies.package.versions).to.eql(['1.0.1', '1.0.0']);
                expect(results.nrDependants).to.equal(0);
                expect(results.versions).to.eql([]);
            });
        });
    });

    it('lists 1 dependency with relative paths when 1 git package installed', function () {
        gitPackage.prepareGit({
            '1.0.0': {
                'bower.json': {
                    name: 'package',
                    main: 'test.txt'
                },
                'version.txt': '1.0.0'
            },
            '1.0.1': {
                'bower.json': {
                    name: 'package',
                    main: 'test2.txt'
                },
                'version.txt': '1.0.1'
            }
        });
        
        tempDir.prepare({
            'bower.json': {
                name: 'test',
                dependencies: {
                    package: gitPackage.path + '#1.0.0'
                }
            }
        });

        return install().then(function() {
            return list({relative: true}).spread(function(results) {
                expect(results.canonicalDir).to.equal(tempDir.path);
                expect(results.pkgMeta.dependencies).to.eql({
                    package: gitPackage.path + '#1.0.0'
                });
                expect(results.dependencies.package.canonicalDir).to.equal(
                    path.normalize('bower_components/package')
                );
            });
        });
    });
});
