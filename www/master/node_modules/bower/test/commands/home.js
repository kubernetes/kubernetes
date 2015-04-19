var Q = require('q');
var expect = require('expect.js');
var helpers = require('../helpers');

var home = helpers.command('home');

describe('bower home', function () {

    it('correctly reads arguments', function() {
        expect(home.readOptions(['foo'])).to.eql(['foo']);
    });

    var package = new helpers.TempDir({
        'bower.json': {
            name: 'package',
            homepage: 'http://bower.io'
        }
    });

    var wrongPackage = new helpers.TempDir({
        'bower.json': {
            name: 'package'
        }
    });

    it('opens repository home page in web browser', function () {
        package.prepare();

        return Q.Promise(function(resolve) {
            var home = helpers.command('home', { opn: resolve });
            helpers.run(home, [package.path]);
        }).then(function(url) {
            expect(url).to.be('http://bower.io');
        });
    });

    it('opens home page of current repository', function () {
        package.prepare();

        return Q.Promise(function(resolve) {
            var home = helpers.command('home', { opn: resolve });
            helpers.run(home, [undefined, { cwd: package.path }]);
        }).then(function(url) {
            expect(url).to.be('http://bower.io');
        });
    });

    it('errors if no homepage is set', function () {
        wrongPackage.prepare();

        return Q.Promise(function(resolve) {
            var home = helpers.command('home', { opn: resolve });
            helpers.run(home, [wrongPackage.path]).fail(resolve);
        }).then(function(reason) {
            expect(reason.message).to.be('No homepage set for package');
            expect(reason.code).to.be('ENOHOME');
        });
    });
});
