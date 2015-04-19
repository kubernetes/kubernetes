var expect = require('expect.js');
var helpers = require('../helpers');

var init = helpers.command('init');

describe('bower init', function () {

    var package = new helpers.TempDir();

    it('correctly reads arguments', function() {
        expect(init.readOptions([]))
        .to.eql([]);
    });

    it('generates bower.json file', function () {
        package.prepare();

        var logger = init({
            cwd: package.path,
            interactive: true
        });

        return helpers.expectEvent(logger, 'prompt')
        .spread(function (prompt, answer) {
            answer({
                name: 'test-name',
                version: 'test-version',
                description: 'test-description',
                moduleType: 'test-moduleType',
                keywords: 'test-keyword',
                authors: 'test-author',
                license: 'test-license',
                homepage: 'test-homepage',
                private: true
            });

            return helpers.expectEvent(logger, 'prompt');
        })
        .spread(function (prompt, answer) {
            answer({ prompt: true });
            return helpers.expectEvent(logger, 'end');
        })
        .then(function () {
            expect(package.readJson('bower.json')).to.eql({
                name: 'test-name',
                version: 'test-version',
                homepage: 'test-homepage',
                authors: [ 'test-author' ],
                description: 'test-description',
                moduleType: 'test-moduleType',
                keywords: [ 'test-keyword' ],
                license: 'test-license'
            });
        });
    });

    it('errors on non-interactive mode', function () {
        package.prepare();

        return helpers.run(init, { cwd: package.path }).then(
            function () { throw 'should fail'; },
            function (reason) {
                expect(reason.message).to.be('Register requires an interactive shell');
                expect(reason.code).to.be('ENOINT');
            }
        );
    });

    it('warns about existing bower.json', function () {
        package.prepare({
            'bower.json': {
                name: 'foobar'
            }
        });

        var logger = init({ cwd: package.path, interactive: true });

        return helpers.expectEvent(logger, 'log').spread(function(event) {
            expect(event.level).to.be('warn');
            expect(event.message).to.be(
                'The existing bower.json file will be used and filled in'
            );
        });
    });
});
